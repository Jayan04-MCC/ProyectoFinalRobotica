/*
 * Deep Q-Learning para GridWorld con CUDA Unified Memory - VERSIÓN CORREGIDA
 * Optimizado para Jetson AGX Xavier (ARM64 + CUDA)
 * Usa Unified Memory para eliminar copias host-device
 * 
 * CORRECCIONES APLICADAS:
 * - Inicialización de pesos mejorada (amplificada x2)
 * - Recompensas intermedias aumentadas (10x)
 * - Gradient clipping menos agresivo (1.0 -> 5.0)
 * - Learning rate decay más lento (0.9995 -> 0.9999)
 * - Epsilon decay más lento (0.998 -> 0.999)
 * - Target update menos frecuente (200 -> 1000 steps)
 * - Pre-fill mejorado (1000 experiencias en vez de 500 episodios)
 * 
 * Compilar: nvcc -o dqn_jetson_fixed dqn_jetson_fixed.cu -O3 -arch=sm_72
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ==================== CONFIGURACIÓN ====================
#define GRID_SIZE 10
#define STATE_SIZE (GRID_SIZE * GRID_SIZE)
#define NUM_ACTIONS 4
#define HIDDEN_SIZE 128
#define BATCH_SIZE 64
#define REPLAY_BUFFER_SIZE 10000
#define LEARNING_RATE 0.001f
#define LR_DECAY 0.9999f  // ← CORREGIDO: decay más lento (era 0.9995)
#define MIN_LR 0.0001f
#define GAMMA 0.95f
#define EPSILON_START 1.0f
#define EPSILON_END 0.1f
#define EPSILON_DECAY 0.999f  // ← CORREGIDO: decay más lento (era 0.998)
#define TARGET_UPDATE_FREQ 1000  // ← CORREGIDO: menos frecuente (era 200)
#define NUM_EPISODES 800
#define MAX_STEPS 50
#define TRAIN_FREQ 4

// ← NUEVO: Gradient clipping más permisivo
#define GRAD_CLIP_THRESHOLD 5.0f  // era 1.0
#define TD_ERROR_CLIP 5.0f  // era 1.0

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        printf("Error code: %d\n", err); \
        exit(1); \
    } \
}

// ==================== ESTRUCTURAS CON UNIFIED MEMORY ====================
struct Experience {
    float state[STATE_SIZE];
    int action;
    float reward;
    float next_state[STATE_SIZE];
    int done;
};

struct ReplayBuffer {
    Experience* buffer;  // Unified Memory
    int size, position, capacity;
};

struct Network {
    float *W1, *b1;  // Unified Memory - accesible desde CPU y GPU
    float *W2, *b2;
    int W1_size, W2_size;
};

struct GridWorld {
    int agent_x, agent_y;
    int goal_x, goal_y;
    int obstacles[GRID_SIZE][GRID_SIZE];
};

// ==================== KERNELS CUDA ====================

// Kernel para capa oculta (STATE -> HIDDEN con ReLU)
__global__ void forward_hidden_kernel(
    const float* W1, const float* b1, const float* states,
    float* hidden, int batch_size
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx < batch_size && hidden_idx < HIDDEN_SIZE) {
        float sum = b1[hidden_idx];
        #pragma unroll 4
        for (int j = 0; j < STATE_SIZE; j++) {
            sum += W1[hidden_idx * STATE_SIZE + j] * states[batch_idx * STATE_SIZE + j];
        }
        hidden[batch_idx * HIDDEN_SIZE + hidden_idx] = fmaxf(0.0f, sum);  // ReLU
    }
}

// Kernel para capa de salida (HIDDEN -> ACTIONS)
__global__ void forward_output_kernel(
    const float* W2, const float* b2, const float* hidden,
    float* output, int batch_size
) {
    int batch_idx = blockIdx.x;
    int action_idx = threadIdx.x;
    
    if (batch_idx < batch_size && action_idx < NUM_ACTIONS) {
        float sum = b2[action_idx];
        #pragma unroll 8
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += W2[action_idx * HIDDEN_SIZE + j] * hidden[batch_idx * HIDDEN_SIZE + j];
        }
        output[batch_idx * NUM_ACTIONS + action_idx] = sum;
    }
}

// Kernel para calcular TD errors con Double DQN - CORREGIDO
__global__ void compute_td_errors_double_dqn_kernel(
    const float* q_policy, const float* q_policy_next, const float* q_target_next,
    const int* actions, const float* rewards, const int* dones,
    float* td_errors, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int action = actions[idx];
        float current_q = q_policy[idx * NUM_ACTIONS + action];
        
        // Double DQN: policy network selecciona la acción
        int best_next_action = 0;
        float max_q_policy = q_policy_next[idx * NUM_ACTIONS];
        for (int a = 1; a < NUM_ACTIONS; a++) {
            float q = q_policy_next[idx * NUM_ACTIONS + a];
            if (q > max_q_policy) {
                max_q_policy = q;
                best_next_action = a;
            }
        }
        
        // Target network evalúa esa acción
        float next_q_value = q_target_next[idx * NUM_ACTIONS + best_next_action];
        
        float target_q = rewards[idx];
        if (!dones[idx]) {
            target_q += GAMMA * next_q_value;
        }
        
        float error = target_q - current_q;
        
        // ← CORREGIDO: Huber loss clipping menos agresivo
        if (error > TD_ERROR_CLIP) error = TD_ERROR_CLIP;
        if (error < -TD_ERROR_CLIP) error = -TD_ERROR_CLIP;
        
        td_errors[idx] = error;
    }
}

// Kernel para gradientes W2 (output layer)
__global__ void compute_grad_W2_kernel(
    const float* hidden, const float* td_errors, const int* actions,
    float* dW2, int batch_size
) {
    int action = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (action < NUM_ACTIONS && hidden_idx < HIDDEN_SIZE) {
        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            if (actions[b] == action) {
                grad_sum += td_errors[b] * hidden[b * HIDDEN_SIZE + hidden_idx];
            }
        }
        atomicAdd(&dW2[action * HIDDEN_SIZE + hidden_idx], grad_sum);
    }
}

// Kernel para gradientes b2
__global__ void compute_grad_b2_kernel(
    const float* td_errors, const int* actions,
    float* db2, int batch_size
) {
    int action = threadIdx.x;
    
    if (action < NUM_ACTIONS) {
        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            if (actions[b] == action) {
                grad_sum += td_errors[b];
            }
        }
        atomicAdd(&db2[action], grad_sum);
    }
}

// Kernel para gradientes W1 (hidden layer)
__global__ void compute_grad_W1_kernel(
    const float* W2, const float* states, const float* hidden,
    const float* td_errors, const int* actions,
    float* dW1, int batch_size
) {
    int hidden_idx = blockIdx.x;
    int state_idx = threadIdx.x;
    
    if (hidden_idx < HIDDEN_SIZE && state_idx < STATE_SIZE) {
        float grad_sum = 0.0f;
        
        for (int b = 0; b < batch_size; b++) {
            float delta = td_errors[b] * W2[actions[b] * HIDDEN_SIZE + hidden_idx];
            // ReLU derivative
            if (hidden[b * HIDDEN_SIZE + hidden_idx] > 0) {
                delta *= states[b * STATE_SIZE + state_idx];
                grad_sum += delta;
            }
        }
        
        atomicAdd(&dW1[hidden_idx * STATE_SIZE + state_idx], grad_sum);
    }
}

// Kernel para gradientes b1
__global__ void compute_grad_b1_kernel(
    const float* W2, const float* hidden,
    const float* td_errors, const int* actions,
    float* db1, int batch_size
) {
    int hidden_idx = threadIdx.x;
    
    if (hidden_idx < HIDDEN_SIZE) {
        float grad_sum = 0.0f;
        
        for (int b = 0; b < batch_size; b++) {
            float delta = td_errors[b] * W2[actions[b] * HIDDEN_SIZE + hidden_idx];
            // ReLU derivative
            if (hidden[b * HIDDEN_SIZE + hidden_idx] > 0) {
                grad_sum += delta;
            }
        }
        
        atomicAdd(&db1[hidden_idx], grad_sum);
    }
}

// Kernel para aplicar gradientes - CORREGIDO
__global__ void apply_gradients_kernel(
    float* weights, const float* gradients, int size, float lr, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = (lr / batch_size) * gradients[idx];
        
        // ← CORREGIDO: Gradient clipping menos agresivo
        if (grad > GRAD_CLIP_THRESHOLD) grad = GRAD_CLIP_THRESHOLD;
        if (grad < -GRAD_CLIP_THRESHOLD) grad = -GRAD_CLIP_THRESHOLD;
        
        weights[idx] += grad;
    }
}

// ==================== FUNCIONES DE RED CON UNIFIED MEMORY ====================

// ← CORREGIDO: Inicialización de pesos mejorada
void init_network(Network* net) {
    net->W1_size = HIDDEN_SIZE * STATE_SIZE;
    net->W2_size = NUM_ACTIONS * HIDDEN_SIZE;
    
    // Unified Memory allocation - accesible desde CPU y GPU
    CUDA_CHECK(cudaMallocManaged(&net->W1, net->W1_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&net->b1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&net->W2, net->W2_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&net->b2, NUM_ACTIONS * sizeof(float)));
    
    // ← CORREGIDO: He initialization con amplificación para redes pequeñas
    // Para redes pequeñas en problemas simples, amplificamos la inicialización
    float std1 = sqrtf(2.0f / STATE_SIZE) * 2.0f;  // Amplificado x2
    float std2 = sqrtf(2.0f / HIDDEN_SIZE) * 1.5f;  // Amplificado x1.5
    
    // Box-Muller para distribución Gaussiana
    for (int i = 0; i < net->W1_size; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        net->W1[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265359f * u2) * std1;
    }
    
    for (int i = 0; i < net->W2_size; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        net->W2[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265359f * u2) * std2;
    }
    
    // ← CORREGIDO: Bias initialization mejorado
    for (int i = 0; i < HIDDEN_SIZE; i++) net->b1[i] = 0.01f;
    for (int i = 0; i < NUM_ACTIONS; i++) net->b2[i] = 0.01f;  // Pequeño bias positivo
    
    printf("[INFO] Inicialización: std1=%.4f, std2=%.4f\n", std1, std2);
}

void free_network(Network* net) {
    cudaFree(net->W1);
    cudaFree(net->b1);
    cudaFree(net->W2);
    cudaFree(net->b2);
}

void copy_network(Network* dst, Network* src) {
    // Copiar directamente en CPU (unified memory hace esto seguro)
    memcpy(dst->W1, src->W1, src->W1_size * sizeof(float));
    memcpy(dst->b1, src->b1, HIDDEN_SIZE * sizeof(float));
    memcpy(dst->W2, src->W2, src->W2_size * sizeof(float));
    memcpy(dst->b2, src->b2, NUM_ACTIONS * sizeof(float));
    
    // Sincronizar para asegurar que GPU vea los cambios
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Forward pass en CPU (unified memory permite acceso directo)
void forward_simple_cpu(Network* net, float* input, float* output) {
    float hidden[HIDDEN_SIZE];
    
    // Hidden layer with ReLU
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = net->b1[i];
        for (int j = 0; j < STATE_SIZE; j++) {
            sum += net->W1[i * STATE_SIZE + j] * input[j];
        }
        hidden[i] = fmaxf(0.0f, sum);
    }
    
    // Output layer
    for (int i = 0; i < NUM_ACTIONS; i++) {
        float sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] = sum;
    }
}

// ==================== GRIDWORLD ====================

void init_gridworld(GridWorld* env) {
    env->goal_x = GRID_SIZE - 1;
    env->goal_y = GRID_SIZE - 1;
    
    memset(env->obstacles, 0, sizeof(env->obstacles));
    env->obstacles[1][1] = 1;
    env->obstacles[2][2] = 1;
    env->obstacles[3][1] = 1;
}

void reset_gridworld(GridWorld* env) {
    env->agent_x = 0;
    env->agent_y = 0;
}

void get_state(GridWorld* env, float* state) {
    memset(state, 0, STATE_SIZE * sizeof(float));
    state[env->agent_y * GRID_SIZE + env->agent_x] = 1.0f;
}

int manhattan_dist(GridWorld* env) {
    return abs(env->agent_x - env->goal_x) + abs(env->agent_y - env->goal_y);
}

// ← CORREGIDO: Recompensas más grandes para mejor señal de gradientes
int step_gridworld(GridWorld* env, int action, float* reward) {
    int old_dist = manhattan_dist(env);
    
    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};
    
    int new_x = env->agent_x + dx[action];
    int new_y = env->agent_y + dy[action];
    
    int moved = 0;
    if (new_x >= 0 && new_x < GRID_SIZE && new_y >= 0 && new_y < GRID_SIZE) {
        if (!env->obstacles[new_y][new_x]) {
            env->agent_x = new_x;
            env->agent_y = new_y;
            moved = 1;
        }
    }
    
    // Goal reached
    if (env->agent_x == env->goal_x && env->agent_y == env->goal_y) {
        *reward = 10.0f;
        return 1;
    }
    
    // ← CORREGIDO: Recompensas intermedias 10x más grandes
    int new_dist = manhattan_dist(env);
    if (new_dist < old_dist) {
        *reward = 1.0f;  // era 0.1
    } else if (new_dist > old_dist) {
        *reward = -2.0f;  // era -0.2
    } else {
        *reward = -0.5f;  // era -0.1
    }
    
    // Penalización extra por chocar con pared/obstáculo
    if (!moved) {
        *reward = -3.0f;  // era -0.3
    }
    
    return 0;
}

void print_grid(GridWorld* env) {
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if (env->agent_x == x && env->agent_y == y) printf(" A ");
            else if (env->goal_x == x && env->goal_y == y) printf(" G ");
            else if (env->obstacles[y][x]) printf(" # ");
            else printf(" . ");
        }
        printf("\n");
    }
}

// ==================== REPLAY BUFFER CON UNIFIED MEMORY ====================

void init_replay_buffer(ReplayBuffer* rb, int capacity) {
    rb->capacity = capacity;
    rb->size = rb->position = 0;
    
    // Unified Memory para replay buffer
    CUDA_CHECK(cudaMallocManaged(&rb->buffer, capacity * sizeof(Experience)));
}

void add_experience(ReplayBuffer* rb, Experience* exp) {
    rb->buffer[rb->position] = *exp;
    rb->position = (rb->position + 1) % rb->capacity;
    if (rb->size < rb->capacity) rb->size++;
}

void free_replay_buffer(ReplayBuffer* rb) {
    cudaFree(rb->buffer);
}

// ==================== DQN TRAINING CON UNIFIED MEMORY ====================

int select_action(Network* net, float* state, float epsilon) {
    if ((float)rand() / RAND_MAX < epsilon) {
        return rand() % NUM_ACTIONS;
    }
    
    float q[NUM_ACTIONS];
    forward_simple_cpu(net, state, q);
    
    int best = 0;
    for (int a = 1; a < NUM_ACTIONS; a++) {
        if (q[a] > q[best]) best = a;
    }
    return best;
}

void train_step_unified(Network* policy, Network* target, ReplayBuffer* rb, float lr) {
    if (rb->size < BATCH_SIZE) return;
    
    // Sample batch (usando unified memory directamente)
    Experience* batch = (Experience*)malloc(BATCH_SIZE * sizeof(Experience));
    for (int b = 0; b < BATCH_SIZE; b++) {
        int idx = rand() % rb->size;
        batch[b] = rb->buffer[idx];
    }
    
    // Preparar datos en unified memory
    float *states, *next_states, *rewards;
    int *actions, *dones;
    
    CUDA_CHECK(cudaMallocManaged(&states, BATCH_SIZE * STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&next_states, BATCH_SIZE * STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&actions, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&rewards, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&dones, BATCH_SIZE * sizeof(int)));
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        memcpy(&states[b * STATE_SIZE], batch[b].state, STATE_SIZE * sizeof(float));
        memcpy(&next_states[b * STATE_SIZE], batch[b].next_state, STATE_SIZE * sizeof(float));
        actions[b] = batch[b].action;
        rewards[b] = batch[b].reward;
        dones[b] = batch[b].done;
    }
    
    // Allocar memoria para forward/backward pass
    float *hidden_policy, *hidden_target, *hidden_policy_next;
    float *q_policy, *q_target, *q_policy_next;
    float *td_errors;
    float *dW1, *db1, *dW2, *db2;
    
    CUDA_CHECK(cudaMallocManaged(&hidden_policy, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&hidden_target, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&hidden_policy_next, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&q_policy, BATCH_SIZE * NUM_ACTIONS * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&q_target, BATCH_SIZE * NUM_ACTIONS * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&q_policy_next, BATCH_SIZE * NUM_ACTIONS * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&td_errors, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&dW1, policy->W1_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&db1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&dW2, policy->W2_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&db2, NUM_ACTIONS * sizeof(float)));
    
    memset(dW1, 0, policy->W1_size * sizeof(float));
    memset(db1, 0, HIDDEN_SIZE * sizeof(float));
    memset(dW2, 0, policy->W2_size * sizeof(float));
    memset(db2, 0, NUM_ACTIONS * sizeof(float));
    
    // Sincronizar antes de usar en GPU
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Forward pass - Policy network (current states)
    forward_hidden_kernel<<<BATCH_SIZE, HIDDEN_SIZE>>>(
        policy->W1, policy->b1, states, hidden_policy, BATCH_SIZE
    );
    forward_output_kernel<<<BATCH_SIZE, NUM_ACTIONS>>>(
        policy->W2, policy->b2, hidden_policy, q_policy, BATCH_SIZE
    );
    
    // Forward pass - Policy network (next states) para Double DQN
    forward_hidden_kernel<<<BATCH_SIZE, HIDDEN_SIZE>>>(
        policy->W1, policy->b1, next_states, hidden_policy_next, BATCH_SIZE
    );
    forward_output_kernel<<<BATCH_SIZE, NUM_ACTIONS>>>(
        policy->W2, policy->b2, hidden_policy_next, q_policy_next, BATCH_SIZE
    );
    
    // Forward pass - Target network (next states)
    forward_hidden_kernel<<<BATCH_SIZE, HIDDEN_SIZE>>>(
        target->W1, target->b1, next_states, hidden_target, BATCH_SIZE
    );
    forward_output_kernel<<<BATCH_SIZE, NUM_ACTIONS>>>(
        target->W2, target->b2, hidden_target, q_target, BATCH_SIZE
    );
    
    // Compute TD errors con Double DQN
    int blocks = (BATCH_SIZE + 255) / 256;
    compute_td_errors_double_dqn_kernel<<<blocks, 256>>>(
        q_policy, q_policy_next, q_target, actions, rewards, dones, td_errors, BATCH_SIZE
    );
    
    // Compute gradients
    compute_grad_W2_kernel<<<NUM_ACTIONS, HIDDEN_SIZE>>>(
        hidden_policy, td_errors, actions, dW2, BATCH_SIZE
    );
    compute_grad_b2_kernel<<<1, NUM_ACTIONS>>>(
        td_errors, actions, db2, BATCH_SIZE
    );
    compute_grad_W1_kernel<<<HIDDEN_SIZE, STATE_SIZE>>>(
        policy->W2, states, hidden_policy, td_errors, actions, dW1, BATCH_SIZE
    );
    compute_grad_b1_kernel<<<1, HIDDEN_SIZE>>>(
        policy->W2, hidden_policy, td_errors, actions, db1, BATCH_SIZE
    );
    
    // Apply gradients
    blocks = (policy->W1_size + 255) / 256;
    apply_gradients_kernel<<<blocks, 256>>>(policy->W1, dW1, policy->W1_size, lr, BATCH_SIZE);
    blocks = (HIDDEN_SIZE + 255) / 256;
    apply_gradients_kernel<<<blocks, 256>>>(policy->b1, db1, HIDDEN_SIZE, lr, BATCH_SIZE);
    blocks = (policy->W2_size + 255) / 256;
    apply_gradients_kernel<<<blocks, 256>>>(policy->W2, dW2, policy->W2_size, lr, BATCH_SIZE);
    apply_gradients_kernel<<<1, NUM_ACTIONS>>>(policy->b2, db2, NUM_ACTIONS, lr, BATCH_SIZE);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup (unified memory se libera automáticamente)
    cudaFree(states); cudaFree(next_states); cudaFree(actions);
    cudaFree(rewards); cudaFree(dones);
    cudaFree(hidden_policy); cudaFree(hidden_target); cudaFree(hidden_policy_next);
    cudaFree(q_policy); cudaFree(q_target); cudaFree(q_policy_next);
    cudaFree(td_errors);
    cudaFree(dW1); cudaFree(db1); cudaFree(dW2); cudaFree(db2);
    
    free(batch);
}

// ==================== MAIN ====================

int main() {
    srand(time(NULL));
    
    printf("=== Deep Q-Learning con CUDA Unified Memory - VERSIÓN CORREGIDA ===\n");
    printf("Grid: %dx%d | Hidden: %d | Batch: %d\n", 
           GRID_SIZE, GRID_SIZE, HIDDEN_SIZE, BATCH_SIZE);
    printf("LR: %.4f -> %.4f (decay: %.5f) | Gamma: %.2f\n",
           LEARNING_RATE, MIN_LR, LR_DECAY, GAMMA);
    printf("Epsilon: %.2f -> %.2f (decay: %.4f)\n",
           EPSILON_START, EPSILON_END, EPSILON_DECAY);
    printf("Target update: %d steps | Grad clip: %.1f | TD clip: %.1f\n\n",
           TARGET_UPDATE_FREQ, GRAD_CLIP_THRESHOLD, TD_ERROR_CLIP);
    
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    int device = 0;
    cudaSetDevice(device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    // Verificar soporte de Unified Memory
    if (!prop.managedMemory) {
        printf("WARNING: Este dispositivo no soporta Unified Memory completamente.\n");
        printf("El código puede funcionar pero con menor eficiencia.\n");
    } else {
        printf("Unified Memory: Soportada\n");
    }
    
    // Verificar concurrent managed access (importante para Jetson)
    if (prop.concurrentManagedAccess) {
        printf("Concurrent Managed Access: Sí (Óptimo para Jetson)\n");
    } else {
        printf("Concurrent Managed Access: No\n");
    }
    printf("\n");
    
    Network policy_net, target_net;
    init_network(&policy_net);
    init_network(&target_net);
    copy_network(&target_net, &policy_net);
    
    ReplayBuffer rb;
    init_replay_buffer(&rb, REPLAY_BUFFER_SIZE);
    
    GridWorld env;
    init_gridworld(&env);
    
    printf("Entorno (A=Agente, G=Meta, #=Obstaculo):\n");
    reset_gridworld(&env);
    print_grid(&env);
    printf("\n");
    
    float epsilon = EPSILON_START;
    float lr = LEARNING_RATE;
    int total_steps = 0;
    float avg_reward = 0;
    float avg_steps = MAX_STEPS;
    int success_history[100] = {0};
    int history_idx = 0;
    int last_100_successes = 0;
    
    // ← CORREGIDO: Pre-fill mejorado - llenar hasta 1000 experiencias
    printf("Llenando replay buffer (target: 1000 experiencias)...\n");
    int target_experiences = 1000;
    int episodes_filled = 0;
    
    while (rb.size < target_experiences) {
        reset_gridworld(&env);
        float state[STATE_SIZE];
        get_state(&env, state);
        
        for (int s = 0; s < MAX_STEPS; s++) {
            int action = rand() % NUM_ACTIONS;
            float reward;
            int done = step_gridworld(&env, action, &reward);
            
            float next_state[STATE_SIZE];
            get_state(&env, next_state);
            
            Experience exp;
            memcpy(exp.state, state, sizeof(exp.state));
            exp.action = action;
            exp.reward = reward;
            memcpy(exp.next_state, next_state, sizeof(exp.next_state));
            exp.done = done;
            add_experience(&rb, &exp);
            
            if (done || rb.size >= target_experiences) break;
            memcpy(state, next_state, sizeof(state));
        }
        episodes_filled++;
    }
    
    // Contar éxitos en buffer inicial
    int initial_successes = 0;
    for (int i = 0; i < rb.size; i++) {
        if (rb.buffer[i].done && rb.buffer[i].reward > 5.0f) {
            initial_successes++;
        }
    }
    
    printf("Buffer: %d experiencias en %d episodios\n", rb.size, episodes_filled);
    printf("Éxitos en buffer inicial: %d (%.1f%%)\n\n", 
           initial_successes, 100.0f * initial_successes / rb.size);
    
    // Training loop
    printf("Iniciando entrenamiento...\n");
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        reset_gridworld(&env);
        float state[STATE_SIZE];
        get_state(&env, state);
        
        float ep_reward = 0;
        int ep_steps = 0;
        int success = 0;
        
        for (int step = 0; step < MAX_STEPS; step++) {
            int action = select_action(&policy_net, state, epsilon);
            
            float reward;
            int done = step_gridworld(&env, action, &reward);
            
            float next_state[STATE_SIZE];
            get_state(&env, next_state);
            
            Experience exp;
            memcpy(exp.state, state, sizeof(exp.state));
            exp.action = action;
            exp.reward = reward;
            memcpy(exp.next_state, next_state, sizeof(exp.next_state));
            exp.done = done;
            add_experience(&rb, &exp);
            
            // Train periodically with Unified Memory
            if (total_steps % TRAIN_FREQ == 0) {
                train_step_unified(&policy_net, &target_net, &rb, lr);
            }
            
            memcpy(state, next_state, sizeof(state));
            ep_reward += reward;
            ep_steps++;
            total_steps++;
            
            // ← CORREGIDO: Update target network menos frecuente
            if (total_steps % TARGET_UPDATE_FREQ == 0) {
                copy_network(&target_net, &policy_net);
                printf("  [Target network actualizada en step %d]\n", total_steps);
            }
            
            if (done) {
                success = 1;
                break;
            }
        }
        
        // Update success history
        last_100_successes -= success_history[history_idx];
        success_history[history_idx] = success;
        last_100_successes += success;
        history_idx = (history_idx + 1) % 100;
        
        // ← CORREGIDO: Decay más lento
        epsilon = fmaxf(EPSILON_END, epsilon * EPSILON_DECAY);
        lr = fmaxf(MIN_LR, lr * LR_DECAY);
        
        avg_reward = 0.95f * avg_reward + 0.05f * ep_reward;
        avg_steps = 0.95f * avg_steps + 0.05f * ep_steps;
        
        if ((episode + 1) % 100 == 0) {
            int window = (episode < 99) ? episode + 1 : 100;
            float success_rate = 100.0f * last_100_successes / window;
            printf("Ep %4d | AvgR: %6.2f | Steps: %5.1f | Eps: %.3f | LR: %.5f | Last100: %.1f%%\n",
                   episode + 1, avg_reward, avg_steps, epsilon, lr, success_rate);
        }
        
        // Early stopping mejorado
        if (episode > 200 && last_100_successes >= 95) {
            printf("\n*** ¡Convergencia alcanzada en episodio %d! ***\n", episode + 1);
            break;
        }
    }
    
    // ==================== EVALUACIÓN ====================
    printf("\n============ EVALUACIÓN ============\n\n");
    
    // Synchronize before CPU evaluation (unified memory se migra automáticamente)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Q-values por celda
    printf("Q-values máximos por celda:\n");
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if (env.obstacles[y][x]) {
                printf("  ##  ");
            } else if (x == env.goal_x && y == env.goal_y) {
                printf(" GOAL ");
            } else {
                float s[STATE_SIZE] = {0};
                s[y * GRID_SIZE + x] = 1.0f;
                float q[NUM_ACTIONS];
                forward_simple_cpu(&policy_net, s, q);
                float maxq = q[0];
                for (int a = 1; a < NUM_ACTIONS; a++)
                    if (q[a] > maxq) maxq = q[a];
                printf("%5.2f ", maxq);
            }
        }
        printf("\n");
    }
    
    printf("\nPolítica (^=UP v=DOWN <=LEFT >=RIGHT):\n");
    const char arrows[] = {'^', 'v', '<', '>'};
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if (env.obstacles[y][x]) {
                printf(" # ");
            } else if (x == env.goal_x && y == env.goal_y) {
                printf(" G ");
            } else {
                float s[STATE_SIZE] = {0};
                s[y * GRID_SIZE + x] = 1.0f;
                float q[NUM_ACTIONS];
                forward_simple_cpu(&policy_net, s, q);
                int best = 0;
                for (int a = 1; a < NUM_ACTIONS; a++)
                    if (q[a] > q[best]) best = a;
                printf(" %c ", arrows[best]);
            }
        }
        printf("\n");
    }
    
    // Demo
    printf("\n--- Demostración ---\n");
    int eval_success = 0;
    for (int trial = 0; trial < 5; trial++) {
        reset_gridworld(&env);
        float state[STATE_SIZE];
        get_state(&env, state);
        
        printf("Trial %d: (0,0)", trial + 1);
        
        for (int step = 0; step < 20; step++) {
            float q[NUM_ACTIONS];
            forward_simple_cpu(&policy_net, state, q);
            
            int action = 0;
            for (int a = 1; a < NUM_ACTIONS; a++)
                if (q[a] > q[action]) action = a;
            
            float reward;
            int done = step_gridworld(&env, action, &reward);
            get_state(&env, state);
            
            printf(" -> (%d,%d)", env.agent_x, env.agent_y);
            
            if (done) {
                printf(" GOAL!\n");
                eval_success++;
                break;
            }
        }
        if (env.agent_x != env.goal_x || env.agent_y != env.goal_y) {
            printf(" (no llegó)\n");
        }
    }
    
    printf("\nResultado: %d/5 éxitos\n", eval_success);
    
    // Cleanup
    free_network(&policy_net);
    free_network(&target_net);
    free_replay_buffer(&rb);
    
    printf("\n=== Entrenamiento completado ===\n");
    
    return 0;
}
