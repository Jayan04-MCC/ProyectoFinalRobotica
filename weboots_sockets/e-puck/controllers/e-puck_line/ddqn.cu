/*
 * Deep Q-Learning para Seguimiento de Línea con e-puck
 * Compilar: nvcc -o dqn_line dqn_line.cu -O3 --shared -Xcompiler -fPIC
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ==================== CONFIGURACIÓN ====================
#define STATE_SIZE 6          // 3 sensores + error + 2 velocidades
#define NUM_ACTIONS 3         // Izquierda, Recto, Derecha
#define HIDDEN1_SIZE 64
#define HIDDEN2_SIZE 32
#define BATCH_SIZE 32
#define REPLAY_BUFFER_SIZE 20000
#define LEARNING_RATE 0.0005f
#define LR_DECAY 0.9998f
#define MIN_LR 0.00005f
#define GAMMA 0.98f
#define EPSILON_START 0.9f
#define EPSILON_END 0.05f
#define EPSILON_DECAY 0.999f
#define TARGET_UPDATE_FREQ 100
#define TRAIN_FREQ 4
#define MAX_STEPS_PER_EPISODE 1000

// Constantes del robot
#define MAX_GS_VALUE 1000.0f
#define MAX_WHEEL_SPEED 1000.0f

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// ==================== ESTRUCTURAS ====================
struct Experience {
    float state[STATE_SIZE];
    int action;
    float reward;
    float next_state[STATE_SIZE];
    int done;
};

struct ReplayBuffer {
    Experience* buffer;
    int size, position, capacity;
};

struct Network {
    float *W1, *b1;  // STATE_SIZE -> HIDDEN1
    float *W2, *b2;  // HIDDEN1 -> HIDDEN2
    float *W3, *b3;  // HIDDEN2 -> NUM_ACTIONS
    int W1_size, W2_size, W3_size;
};

// Variables globales para el agente
static Network policy_net, target_net;
static ReplayBuffer replay_buffer;
static float epsilon = EPSILON_START;
static float learning_rate = LEARNING_RATE;
static int total_steps = 0;
static int episode_count = 0;
static int training_enabled = 1;

// ==================== FUNCIONES DE RED ====================

void init_network(Network* net) {
    net->W1_size = HIDDEN1_SIZE * STATE_SIZE;
    net->W2_size = HIDDEN2_SIZE * HIDDEN1_SIZE;
    net->W3_size = NUM_ACTIONS * HIDDEN2_SIZE;
    
    net->W1 = (float*)malloc(net->W1_size * sizeof(float));
    net->b1 = (float*)malloc(HIDDEN1_SIZE * sizeof(float));
    net->W2 = (float*)malloc(net->W2_size * sizeof(float));
    net->b2 = (float*)malloc(HIDDEN2_SIZE * sizeof(float));
    net->W3 = (float*)malloc(net->W3_size * sizeof(float));
    net->b3 = (float*)malloc(NUM_ACTIONS * sizeof(float));
    
    // He initialization
    float std1 = sqrtf(2.0f / STATE_SIZE);
    float std2 = sqrtf(2.0f / HIDDEN1_SIZE);
    float std3 = sqrtf(2.0f / HIDDEN2_SIZE);
    
    for (int i = 0; i < net->W1_size; i++) {
        float u1 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
        float u2 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
        net->W1[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2) * std1;
    }
    for (int i = 0; i < net->W2_size; i++) {
        float u1 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
        float u2 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
        net->W2[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2) * std2;
    }
    for (int i = 0; i < net->W3_size; i++) {
        float u1 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
        float u2 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
        net->W3[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2) * std3;
    }
    
    for (int i = 0; i < HIDDEN1_SIZE; i++) net->b1[i] = 0.01f;
    for (int i = 0; i < HIDDEN2_SIZE; i++) net->b2[i] = 0.01f;
    for (int i = 0; i < NUM_ACTIONS; i++) net->b3[i] = 0.0f;
}

void free_network(Network* net) {
    free(net->W1); free(net->b1);
    free(net->W2); free(net->b2);
    free(net->W3); free(net->b3);
}

void copy_network(Network* dst, Network* src) {
    memcpy(dst->W1, src->W1, src->W1_size * sizeof(float));
    memcpy(dst->b1, src->b1, HIDDEN1_SIZE * sizeof(float));
    memcpy(dst->W2, src->W2, src->W2_size * sizeof(float));
    memcpy(dst->b2, src->b2, HIDDEN2_SIZE * sizeof(float));
    memcpy(dst->W3, src->W3, src->W3_size * sizeof(float));
    memcpy(dst->b3, src->b3, NUM_ACTIONS * sizeof(float));
}

// Forward pass: Input -> Hidden1 -> Hidden2 -> Output
void forward(Network* net, float* input, float* h1, float* h2, float* output) {
    // Capa 1: input -> h1 (ReLU)
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        float sum = net->b1[i];
        for (int j = 0; j < STATE_SIZE; j++) {
            sum += net->W1[i * STATE_SIZE + j] * input[j];
        }
        h1[i] = fmaxf(0.0f, sum);
    }
    
    // Capa 2: h1 -> h2 (ReLU)
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        float sum = net->b2[i];
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            sum += net->W2[i * HIDDEN1_SIZE + j] * h1[j];
        }
        h2[i] = fmaxf(0.0f, sum);
    }
    
    // Capa 3: h2 -> output (lineal)
    for (int i = 0; i < NUM_ACTIONS; i++) {
        float sum = net->b3[i];
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            sum += net->W3[i * HIDDEN2_SIZE + j] * h2[j];
        }
        output[i] = sum;
    }
}

void forward_simple(Network* net, float* input, float* output) {
    float h1[HIDDEN1_SIZE], h2[HIDDEN2_SIZE];
    forward(net, input, h1, h2, output);
}

// ==================== REPLAY BUFFER ====================

void init_replay_buffer(ReplayBuffer* rb, int capacity) {
    rb->buffer = (Experience*)malloc(capacity * sizeof(Experience));
    rb->size = 0;
    rb->position = 0;
    rb->capacity = capacity;
}

void add_experience(ReplayBuffer* rb, Experience* exp) {
    rb->buffer[rb->position] = *exp;
    rb->position = (rb->position + 1) % rb->capacity;
    if (rb->size < rb->capacity) rb->size++;
}

void free_replay_buffer(ReplayBuffer* rb) {
    free(rb->buffer);
}

// ==================== SISTEMA DE RECOMPENSAS ====================

float calculate_reward(float state[STATE_SIZE], float next_state[STATE_SIZE]) {
    float reward = 0.0f;
    
    // 1. Recompensa por mantener el centro sobre la línea
    float center_intensity = next_state[1];  // gs_center normalizado
    if (center_intensity < 0.5f) {  // Sobre línea negra
        reward += 1.0f;
    } else {
        reward -= 0.5f;  // Fuera de la línea
    }
    
    // 2. Penalización por error lateral
    float error = fabs(next_state[3]);  // Error normalizado
    reward -= error * 1.5f;
    
    // 3. Recompensa por reducir el error
    float prev_error = fabs(state[3]);
    if (error < prev_error) {
        reward += 0.3f;
    }
    
    // 4. Penalización severa por perder completamente la línea
    float avg_sensors = (next_state[0] + next_state[1] + next_state[2]) / 3.0f;
    if (avg_sensors > 0.95f) {  // Todos ven blanco
        reward -= 10.0f;
    }
    
    // 5. Pequeña recompensa por mantener velocidad
    float avg_speed = (fabs(next_state[4]) + fabs(next_state[5])) / 2.0f;
    if (avg_speed > 0.4f) {  // Velocidad razonable
        reward += 0.1f;
    }
    
    return reward;
}

int is_done(float state[STATE_SIZE], int step_count) {
    // Perdió completamente la línea
    float avg_sensors = (state[0] + state[1] + state[2]) / 3.0f;
    if (avg_sensors > 0.95f) {
        return 1;
    }
    
    // Máximo de pasos por episodio
    if (step_count >= MAX_STEPS_PER_EPISODE) {
        return 1;
    }
    
    return 0;
}

// ==================== SELECCIÓN DE ACCIÓN ====================

int select_action(Network* net, float* state, float eps) {
    if ((float)rand() / RAND_MAX < eps) {
        return rand() % NUM_ACTIONS;
    }
    
    float q[NUM_ACTIONS];
    forward_simple(net, state, q);
    
    int best = 0;
    for (int a = 1; a < NUM_ACTIONS; a++) {
        if (q[a] > q[best]) best = a;
    }
    return best;
}

// ==================== ENTRENAMIENTO ====================

void train_step(Network* policy, Network* target, ReplayBuffer* rb, float lr) {
    if (rb->size < BATCH_SIZE) return;
    
    // Gradientes
    float* dW1 = (float*)calloc(policy->W1_size, sizeof(float));
    float* db1 = (float*)calloc(HIDDEN1_SIZE, sizeof(float));
    float* dW2 = (float*)calloc(policy->W2_size, sizeof(float));
    float* db2 = (float*)calloc(HIDDEN2_SIZE, sizeof(float));
    float* dW3 = (float*)calloc(policy->W3_size, sizeof(float));
    float* db3 = (float*)calloc(NUM_ACTIONS, sizeof(float));
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        int idx = rand() % rb->size;
        Experience* exp = &rb->buffer[idx];
        
        // Forward policy network
        float h1[HIDDEN1_SIZE], h2[HIDDEN2_SIZE], q[NUM_ACTIONS];
        forward(policy, exp->state, h1, h2, q);
        
        // Double DQN
        float q_next_policy[NUM_ACTIONS];
        forward_simple(policy, exp->next_state, q_next_policy);
        int best_next_action = 0;
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (q_next_policy[a] > q_next_policy[best_next_action]) 
                best_next_action = a;
        }
        
        float q_next_target[NUM_ACTIONS];
        forward_simple(target, exp->next_state, q_next_target);
        float next_q_value = q_next_target[best_next_action];
        
        // Target Q-value
        float target_q = exp->reward;
        if (!exp->done) {
            target_q += GAMMA * next_q_value;
        }
        
        // TD Error con clipping
        float td_error = target_q - q[exp->action];
        float grad = fminf(fmaxf(td_error, -1.0f), 1.0f);
        
        // === BACKPROP ===
        // Gradiente capa 3 (output)
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            dW3[exp->action * HIDDEN2_SIZE + j] += grad * h2[j];
        }
        db3[exp->action] += grad;
        
        // Gradiente capa 2
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            float delta = grad * policy->W3[exp->action * HIDDEN2_SIZE + j];
            delta *= (h2[j] > 0) ? 1.0f : 0.0f;
            
            for (int k = 0; k < HIDDEN1_SIZE; k++) {
                dW2[j * HIDDEN1_SIZE + k] += delta * h1[k];
            }
            db2[j] += delta;
        }
        
        // Gradiente capa 1
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            float delta = 0.0f;
            for (int i = 0; i < HIDDEN2_SIZE; i++) {
                float delta2 = grad * policy->W3[exp->action * HIDDEN2_SIZE + i];
                delta2 *= (h2[i] > 0) ? 1.0f : 0.0f;
                delta += delta2 * policy->W2[i * HIDDEN1_SIZE + j];
            }
            delta *= (h1[j] > 0) ? 1.0f : 0.0f;
            
            for (int k = 0; k < STATE_SIZE; k++) {
                dW1[j * STATE_SIZE + k] += delta * exp->state[k];
            }
            db1[j] += delta;
        }
    }
    
    // Aplicar gradientes
    float scale = lr / BATCH_SIZE;
    for (int i = 0; i < policy->W1_size; i++) policy->W1[i] += scale * dW1[i];
    for (int i = 0; i < HIDDEN1_SIZE; i++) policy->b1[i] += scale * db1[i];
    for (int i = 0; i < policy->W2_size; i++) policy->W2[i] += scale * dW2[i];
    for (int i = 0; i < HIDDEN2_SIZE; i++) policy->b2[i] += scale * db2[i];
    for (int i = 0; i < policy->W3_size; i++) policy->W3[i] += scale * dW3[i];
    for (int i = 0; i < NUM_ACTIONS; i++) policy->b3[i] += scale * db3[i];
    
    free(dW1); free(db1); free(dW2); free(db2); free(dW3); free(db3);
}

// ==================== FUNCIONES DE GUARDADO/CARGA ====================

void save_network(Network* net, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: No se pudo abrir %s para escritura\n", filename);
        return;
    }
    
    fwrite(&net->W1_size, sizeof(int), 1, f);
    fwrite(&net->W2_size, sizeof(int), 1, f);
    fwrite(&net->W3_size, sizeof(int), 1, f);
    
    fwrite(net->W1, sizeof(float), net->W1_size, f);
    fwrite(net->b1, sizeof(float), HIDDEN1_SIZE, f);
    fwrite(net->W2, sizeof(float), net->W2_size, f);
    fwrite(net->b2, sizeof(float), HIDDEN2_SIZE, f);
    fwrite(net->W3, sizeof(float), net->W3_size, f);
    fwrite(net->b3, sizeof(float), NUM_ACTIONS, f);
    
    fclose(f);
    printf("Red guardada en %s\n", filename);
}

void load_network(Network* net, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Advertencia: No se pudo abrir %s, usando pesos aleatorios\n", filename);
        return;
    }
    
    int w1_size, w2_size, w3_size;
    fread(&w1_size, sizeof(int), 1, f);
    fread(&w2_size, sizeof(int), 1, f);
    fread(&w3_size, sizeof(int), 1, f);
    
    if (w1_size != net->W1_size || w2_size != net->W2_size || w3_size != net->W3_size) {
        printf("Error: Tamaño de red incompatible en %s\n", filename);
        fclose(f);
        return;
    }
    
    fread(net->W1, sizeof(float), net->W1_size, f);
    fread(net->b1, sizeof(float), HIDDEN1_SIZE, f);
    fread(net->W2, sizeof(float), net->W2_size, f);
    fread(net->b2, sizeof(float), HIDDEN2_SIZE, f);
    fread(net->W3, sizeof(float), net->W3_size, f);
    fread(net->b3, sizeof(float), NUM_ACTIONS, f);
    
    fclose(f);
    printf("Red cargada desde %s\n", filename);
}

// ==================== API EXTERNA (para Webots) ====================

extern "C" {

void ddqn_initialize() {
    srand(time(NULL));
    
    printf("=== Inicializando Deep Q-Learning para Line Following ===\n");
    printf("Estado: %d | Acciones: %d | Red: %d->%d->%d->%d\n",
           STATE_SIZE, NUM_ACTIONS, STATE_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, NUM_ACTIONS);
    
    init_network(&policy_net);
    init_network(&target_net);
    
    // Intentar cargar pesos guardados
    load_network(&policy_net, "ddqn_line_policy.bin");
    copy_network(&target_net, &policy_net);
    
    init_replay_buffer(&replay_buffer, REPLAY_BUFFER_SIZE);
    
    epsilon = EPSILON_START;
    learning_rate = LEARNING_RATE;
    total_steps = 0;
    episode_count = 0;
    
    printf("Inicialización completa. Epsilon: %.3f, LR: %.5f\n", epsilon, learning_rate);
}

int ddqn_predict(double state_double[STATE_SIZE]) {
    // Convertir double a float
    float state[STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = (float)state_double[i];
    }
    
    return select_action(&policy_net, state, epsilon);
}

void ddqn_update(double state_double[STATE_SIZE], int action, float reward,
                 double next_state_double[STATE_SIZE], int done) {
    
    if (!training_enabled) return;
    
    // Convertir a float
    float state[STATE_SIZE], next_state[STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = (float)state_double[i];
        next_state[i] = (float)next_state_double[i];
    }
    
    // Guardar experiencia
    Experience exp;
    memcpy(exp.state, state, sizeof(exp.state));
    exp.action = action;
    exp.reward = reward;
    memcpy(exp.next_state, next_state, sizeof(exp.next_state));
    exp.done = done;
    add_experience(&replay_buffer, &exp);
    
    // Entrenar
    if (total_steps % TRAIN_FREQ == 0 && replay_buffer.size >= BATCH_SIZE) {
        train_step(&policy_net, &target_net, &replay_buffer, learning_rate);
    }
    
    total_steps++;
    
    // Actualizar target network
    if (total_steps % TARGET_UPDATE_FREQ == 0) {
        copy_network(&target_net, &policy_net);
        printf("[Step %d] Target network actualizada | Buffer: %d | Epsilon: %.3f\n",
               total_steps, replay_buffer.size, epsilon);
    }
    
    // Decay
    epsilon = fmaxf(EPSILON_END, epsilon * EPSILON_DECAY);
    learning_rate = fmaxf(MIN_LR, learning_rate * LR_DECAY);
    
    // Al final de episodio
    if (done) {
        episode_count++;
        if (episode_count % 10 == 0) {
            printf("Episodio %d completo | Steps: %d | Epsilon: %.3f | LR: %.5f\n",
                   episode_count, total_steps, epsilon, learning_rate);
        }
        
        // Guardar periódicamente
        if (episode_count % 50 == 0) {
            save_network(&policy_net, "ddqn_line_policy.bin");
        }
    }
}

void ddqn_save_model(const char* filename) {
    save_network(&policy_net, filename);
}

void ddqn_load_model(const char* filename) {
    load_network(&policy_net, filename);
    copy_network(&target_net, &policy_net);
}

void ddqn_set_training(int enabled) {
    training_enabled = enabled;
    printf("Entrenamiento %s\n", enabled ? "ACTIVADO" : "DESACTIVADO");
}

void ddqn_cleanup() {
    free_network(&policy_net);
    free_network(&target_net);
    free_replay_buffer(&replay_buffer);
    printf("Recursos liberados\n");
}

float ddqn_get_epsilon() {
    return epsilon;
}

void ddqn_set_epsilon(float eps) {
    epsilon = eps;
}

} // extern "C"

// ==================== MAIN (para testing standalone) ====================

int main() {
    printf("=== Test Standalone del DQN Line Following ===\n");
    
    ddqn_initialize();
    
    // Simular algunos pasos
    double state[STATE_SIZE] = {0.3, 0.2, 0.8, 0.5, 0.5, 0.5};  // Ejemplo
    
    for (int i = 0; i < 10; i++) {
        int action = ddqn_predict(state);
        printf("Estado: [%.2f, %.2f, %.2f] Error: %.2f -> Acción: %d\n",
               state[0], state[1], state[2], state[3], action);
        
        // Simular transición
        double next_state[STATE_SIZE];
        memcpy(next_state, state, sizeof(state));
        next_state[1] += (action == 1) ? 0.1 : -0.05;  // Mejorar si va recto
        
        float reward = (float)((state[1] < 0.5) ? 1.0 : -0.5);
        int done = 0;
        
        ddqn_update(state, action, reward, next_state, done);
        memcpy(state, next_state, sizeof(state));
    }
    
    ddqn_save_model("test_model.bin");
    ddqn_cleanup();
    
    printf("\nTest completado\n");
    return 0;
}