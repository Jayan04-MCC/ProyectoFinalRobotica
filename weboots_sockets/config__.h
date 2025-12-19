// config.h
#ifndef CONFIG_H
#define CONFIG_H

// Hiperparámetros
#define STATE_SIZE 6
#define NUM_ACTIONS 3
#define HIDDEN_SIZE 256
#define BATCH_SIZE 64
#define REPLAY_BUFFER_SIZE 50000
#define LEARNING_RATE 0.001f 
#define LR_DECAY 0.99999f 
#define MIN_LR 0.0001f
#define GAMMA 0.95f
#define EPSILON_START 1.0f
#define EPSILON_END 0.05f
#define EPSILON_DECAY 0.9998f 
#define TARGET_UPDATE_FREQ 500
#define NUM_EPISODES 2000
#define MAX_STEPS 300
#define TRAIN_FREQ 2

// Límites físicos
#define MAX_GS_VALUE 1000.0f
#define MAX_WHEEL_SPEED 550.0f

// Índices del estado
#define GS_LEFT 0
#define GS_CENTER 1
#define GS_RIGHT 2
#define GS_LEFT_SPEED 3
#define GS_RIGHT_SPEED 4

// Clipping
#define GRAD_CLIP_THRESHOLD 5.0f
#define TD_ERROR_CLIP 5.0f

// Macros útiles
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        printf("Error code: %d\n", err); \
        exit(1); \
    } \
}

#endif // CONFIG_H