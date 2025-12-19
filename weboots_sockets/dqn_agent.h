// dqn_agent.h
#ifndef DQN_AGENT_H
#define DQN_AGENT_H

#include "types.h"
#include <vector>

// Red neuronal
void init_network(Network* net);
void free_network(Network* net);
void copy_network(Network* dst, Network* src);
void forward_simple_cpu(Network* net, float* input, float* output);

// Replay buffer
void init_replay_buffer(ReplayBuffer* rb, int capacity);
void add_experience(ReplayBuffer* rb, Experience* exp);
void free_replay_buffer(ReplayBuffer* rb);

// Entrenamiento
int select_action(Network* net, float* state, float epsilon);
void train_step_unified(Network* policy, Network* target, ReplayBuffer* rb, float lr);

// Recompensa y estado
float compute_reward(const std::vector<int>& data);
int compute_done(const std::vector<int>& data, int step_count);
void get_state(float state[STATE_SIZE], const std::vector<int>& data);

// Métricas
void print_detailed_metrics(int episode, float ep_reward, int ep_steps,
                            float epsilon, float lr, int success);

// Socket
bool sendAll(int sock, const void* data, size_t size);
bool recvAll(int sock, void* data, size_t size);

#endif // DQN_AGENT_H