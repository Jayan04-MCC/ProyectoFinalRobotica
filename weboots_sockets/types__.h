// types.h
#ifndef TYPES_H
#define TYPES_H

#include "config.h"
#include <vector>

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
    float *W1, *b1;
    float *W2, *b2;
    int W1_size, W2_size;
};

#endif // TYPES_H