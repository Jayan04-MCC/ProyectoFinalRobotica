/*
 * Deep Q-Learning para GridWorld con CUDA Unified Memory
 * Optimizado para Jetson AGX Xavier (ARM64 + CUDA)
 * Usa Unified Memory para eliminar copias host-device
 */

#include "config.h"
#include "types.h"
#include "cuda_kernels.cuh"
#include "dqn_agent.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

void print_detailed_metrics(int episode, float ep_reward, int ep_steps, 
                            float epsilon, float lr, int success) {
    static float best_reward = -1000.0f;
    static int consecutive_improvements = 0;
    
    if (ep_reward > best_reward) {
        best_reward = ep_reward;
        consecutive_improvements++;
    } else {
        consecutive_improvements = 0;
    }
    
    if ((episode + 1) % 10 == 0) {
        printf("Ep %4d | Reward %7.2f (Best: %7.2f) | Steps %3d | "
               "Eps %.3f | LR %.6f | Success %d | Improvements %d\n",
               episode + 1, ep_reward, best_reward, ep_steps,
               epsilon, lr, success, consecutive_improvements);
    }
    
    // Advertencia si no mejora en 100 episodios
    if (consecutive_improvements == 0 && episode > 100 && episode % 100 == 0) {
        printf("WARNING: No improvement in last 100 episodes!\n");
    }
}

// ==================== FUNCIONES DE RED CON UNIFIED MEMORY ====================
void init_network(Network* net) {
    net->W1_size = HIDDEN_SIZE * STATE_SIZE;
    net->W2_size = NUM_ACTIONS * HIDDEN_SIZE;

    // Unified Memory allocation - accesible desde CPU y GPU
    CUDA_CHECK(cudaMallocManaged(&net->W1, net->W1_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&net->b1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&net->W2, net->W2_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&net->b2, NUM_ACTIONS * sizeof(float)));

    float std1 = sqrtf(2.0f / STATE_SIZE) * 2.0f;
    float std2 = sqrtf(2.0f / HIDDEN_SIZE) * 1.5f;

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
    memcpy(dst->W1, src->W1, src->W1_size * sizeof(float));
    memcpy(dst->b1, src->b1, HIDDEN_SIZE * sizeof(float));
    memcpy(dst->W2, src->W2, src->W2_size * sizeof(float));
    memcpy(dst->b2, src->b2, NUM_ACTIONS * sizeof(float));
    // Sincronizar para asegurar que GPU vea los cambios
    CUDA_CHECK(cudaDeviceSynchronize());
}

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


// ===================================================
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

float compute_reward(const std::vector<int>& data) {
    int gs_value[3];
    gs_value[0] = data[GS_LEFT];
    gs_value[1] = data[GS_CENTER];
    gs_value[2] = data[GS_RIGHT];
    int speed_left = data[GS_LEFT_SPEED];
    int speed_right = data[GS_RIGHT_SPEED];
    
    // Normalizar sensores (negro=1.0, blanco=0.0)
    float left_norm = 1.0f - (gs_value[GS_LEFT] / MAX_GS_VALUE);
    float center_norm = 1.0f - (gs_value[GS_CENTER] / MAX_GS_VALUE);
    float right_norm = 1.0f - (gs_value[GS_RIGHT] / MAX_GS_VALUE);
    float reward = 0.0f;
    
    if (center_norm > 0.5f) {  // Centro detecta negro
        reward += 2.0f;
    }
    
    // B. Bonificación por balance simétrico (robot centrado)
    float balance = 1.0f - fabsf(left_norm - right_norm);
    reward += balance * 1.5f;
    
    // C. Penalización por desviación lateral
    float lateral_error = fabsf(left_norm - right_norm);
    reward -= lateral_error * 0.5f;
    
    // D. Bonificación por avanzar (velocidad promedio alta)
    float avg_speed = (speed_left + speed_right) / (2.0f * MAX_WHEEL_SPEED);
    if (center_norm > 0.5f) {  // Solo si sigue la línea
        reward += avg_speed * 1.0f;
    }
    
    // E. Penalización moderada por oscilación
    float speed_diff = fabsf((float)speed_left - (float)speed_right) / MAX_WHEEL_SPEED;
    reward -= speed_diff * 0.2f;
    
    // ── PENALIZACIONES FUERTES ──
    
    // F. Pérdida total de línea (todos los sensores en blanco)
    if (left_norm < 0.2f && center_norm < 0.2f && right_norm < 0.2f) {
        reward = -10.0f;  // Penalización catastrófica
    }
    
    // G. Bonificación por estado ideal (todos sobre negro)
    if (left_norm > 0.7f && center_norm > 0.7f && right_norm > 0.7f) {
        reward += 3.0f;
    }
    
    return reward;
}

int compute_done(const std::vector<int>& data, int step_count) {
    int gs_value[3];
    gs_value[0] = data[GS_LEFT];
    gs_value[1] = data[GS_CENTER];
    gs_value[2] = data[GS_RIGHT];
    
    // 1. Línea completamente perdida (todo blanco)
    if (gs_value[0] > 900 && gs_value[1] > 900 && gs_value[2] > 900) {
        return 1;
    }
    
    // 2. Robot perfectamente sobre la línea por tiempo prolongado
    static int perfect_steps = 0;
    if (gs_value[0] < 300 && gs_value[1] < 300 && gs_value[2] < 300) {
        perfect_steps++;
        if (perfect_steps >= 50) {  // 50 steps consecutivos perfecto
            perfect_steps = 0;
            return 1;  // Episodio exitoso
        }
    } else {
        perfect_steps = 0;
    }
    
    // 3. Timeout
    if (step_count >= MAX_STEPS) {
        perfect_steps = 0;
        return 1;
    }
    
    return 0;
}
void get_state(float state[STATE_SIZE], const std::vector<int>& data) {
    state[0] = 1.0f - (data[0] / MAX_GS_VALUE);  // GS_LEFT
    state[1] = 1.0f - (data[1] / MAX_GS_VALUE);  // GS_CENTER
    state[2] = 1.0f - (data[2] / MAX_GS_VALUE);  // GS_RIGHT
    
    // Error lateral (positivo = robot desviado a la derecha)
    state[3] = ((float)data[0] - (float)data[2]) / MAX_GS_VALUE;
    
    // Velocidades normalizadas
    state[4] = data[3] / MAX_WHEEL_SPEED;  // speed_left
    state[5] = data[4] / MAX_WHEEL_SPEED;  // speed_right
}


// ====== SOCKET UTILS ======
bool sendAll(int sock, const void* data, size_t size) {
    size_t sent = 0;
    while (sent < size) {
        ssize_t n = send(sock, (const char*)data + sent, size - sent, 0);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}
bool recvAll(int sock, void* data, size_t size) {
    size_t recvd = 0;
    while (recvd < size) {
        ssize_t n = recv(sock, (char*)data + recvd, size - recvd, 0);
        if (n <= 0) return false;
        recvd += n;
    }
    return true;
}


int main() {
    srand(time(NULL));

    // ================= CUDA INIT =================
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    cudaSetDevice(0);

    Network policy_net, target_net;
    init_network(&policy_net);
    init_network(&target_net);
    copy_network(&target_net, &policy_net);

    ReplayBuffer rb;
    init_replay_buffer(&rb, REPLAY_BUFFER_SIZE);

    float epsilon = EPSILON_START;
    float lr = LEARNING_RATE;
    int total_steps = 0;
    float avg_reward = 0;
    float avg_steps = MAX_STEPS;

    int success_history[100] = {0};
    int history_idx = 0;
    int last_100_successes = 0;

    // ================= SOCKET SERVER =================
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket failed");
        return -1;
    }

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_conn.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(5000);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind failed");
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 1) < 0) {
        perror("listen failed");
        close(server_fd);
        return -1;
    }

    printf("[TRAINER] Esperando conexión...\n");
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int peer_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (peer_fd < 0) {
        perror("accept failed");
        close(server_fd);
        return -1;
    }
    printf("[TRAINER] ENV conectado\n");

    // ================= LOOP PRINCIPAL =================
    int episode = 0;
    while (episode < NUM_EPISODES) {
        // -------- ESPERAR DATOS CRUDOS --------
        uint32_t data_size;
        if (!recvAll(peer_fd, &data_size, sizeof(data_size)))
            break;
        data_size = ntohl(data_size);
        std::vector<int> data(data_size);
        if (!recvAll(peer_fd, data.data(), data_size * sizeof(int)))
            break;

        // -------- ENTRENAR UNA ÉPOCA --------
        float state[STATE_SIZE];
        get_state(state, data);
        float ep_reward = 0;
        int ep_steps = 0;
        int success = 0;
        int action = 0;

        for (int step = 0; step < MAX_STEPS; step++) {
            action = select_action(&policy_net, state, epsilon);
            int done = compute_done(data, step);
            float reward = compute_reward(data);
            float next_state[STATE_SIZE];
            get_state(next_state, data);

            Experience exp;
            memcpy(exp.state, state, sizeof(exp.state));
            exp.action = action;
            exp.reward = reward;
            memcpy(exp.next_state, next_state, sizeof(exp.next_state));
            exp.done = done;

            add_experience(&rb, &exp);

            if (total_steps % TRAIN_FREQ == 0) {
                train_step_unified(&policy_net, &target_net, &rb, lr);
            }

            memcpy(state, next_state, sizeof(state));
            ep_reward += reward;
            ep_steps++;
            total_steps++;

            if (total_steps % TARGET_UPDATE_FREQ == 0) {
                copy_network(&target_net, &policy_net);
            }

            if (done) {
                success = data[GS_CENTER] < 450;
                break;
            }
        }

        // -------- MÉTRICAS --------
        last_100_successes -= success_history[history_idx];
        success_history[history_idx] = success;
        last_100_successes += success;
        history_idx = (history_idx + 1) % 100;

        epsilon = fmaxf(EPSILON_END, epsilon * EPSILON_DECAY);
        lr = fmaxf(MIN_LR, lr * LR_DECAY);
        avg_reward = 0.95f * avg_reward + 0.05f * ep_reward;
        avg_steps = 0.95f * avg_steps + 0.05f * ep_steps;

        if ((episode + 1) % 10 == 0) {
            int window = (episode < 99) ? episode + 1 : 100;
            float success_rate = 100.0f * last_100_successes / window;
            printf("Ep %4d | AvgR %.2f | Steps %.1f | Eps %.3f | Last100 %.1f%%\n",
                   episode + 1, avg_reward, avg_steps, epsilon, success_rate);
        }

        // -------- RESPONDER ACCIÓN --------
        int32_t net_action = htonl(action);
        if (!sendAll(peer_fd, &net_action, sizeof(net_action)))
            break;

        episode++;

        if (episode > 200 && last_100_successes >= 95) {
            printf("*** Convergencia alcanzada ***\n");
            break;
        }

        print_detailed_metrics(episode, ep_reward, ep_steps, epsilon, lr, success);
    }

    // ================= CLEANUP =================
    close(peer_fd);
    close(server_fd);
    free_network(&policy_net);
    free_network(&target_net);
    free_replay_buffer(&rb);
    printf("[TRAINER] Finalizado\n");

    return 0;
}