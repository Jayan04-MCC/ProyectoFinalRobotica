// cuda_kernels.cuh
#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "type.h"

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
        if (grad > GRAD_CLIP_THRESHOLD) grad = GRAD_CLIP_THRESHOLD;
        if (grad < -GRAD_CLIP_THRESHOLD) grad = -GRAD_CLIP_THRESHOLD;
        weights[idx] += grad;
    }
}

#endif // CUDA_KERNELS_CUH