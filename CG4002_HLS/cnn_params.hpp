#ifndef CNN_PARAMS_HPP
#define CNN_PARAMS_HPP

#include "cnn_basic_ops.hpp"

// Input Norm Params
extern const data_t input_mean[9];
extern const data_t input_inv_std[9];

// Layer 1
extern const data_t conv1_weight[32][9][5];
extern const data_t conv1_bias[32];
extern const data_t bn1_gamma[32];
extern const data_t bn1_beta[32];
extern const data_t bn1_mean[32];
extern const data_t bn1_inv_std[32];

// Layer 2
extern const data_t conv2_weight[64][32][5];
extern const data_t conv2_bias[64];
extern const data_t bn2_gamma[64];
extern const data_t bn2_beta[64];
extern const data_t bn2_mean[64];
extern const data_t bn2_inv_std[64];

// Layer 3
extern const data_t conv3_weight[128][64][3];
extern const data_t conv3_bias[128];
extern const data_t bn3_gamma[128];
extern const data_t bn3_beta[128];
extern const data_t bn3_mean[128];
extern const data_t bn3_inv_std[128];

// Classifier
extern const data_t fc_weight[6][128];
extern const data_t fc_bias[6];

#endif