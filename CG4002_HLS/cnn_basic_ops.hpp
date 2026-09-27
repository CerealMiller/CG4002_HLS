#ifndef CNN_BASIC_OPS_H
#define CNN_BASIC_OPS_H

#include <ap_fixed.h>

// Using ap_fixed for HLS optimization (16-bit word, 8-bit integer)
typedef ap_fixed<16, 8> data_t;
typedef ap_fixed<32, 12> acc_t;

// ReLU
data_t relu(data_t x);
void relu_1d_128(const data_t in[128], data_t out[128]);
void relu_2d_32x50(const data_t in[32][50], data_t out[32][50]);
void relu_2d_64x25(const data_t in[64][25], data_t out[64][25]);
void relu_2d_128x12(const data_t in[128][12], data_t out[128][12]);

// Normalization
void normalize_9(
    const data_t in[9],
    const data_t mean[9],
    const data_t inv_std[9], // Fixed: changed from scalar to array
    data_t out[9]
);

void normalize_9x50(
    const data_t in[9][50],
    const data_t mean[9],
    const data_t inv_std[9],
    data_t out[9][50]
);

void batchnorm1d_32x50(
    const data_t in[32][50],
    const data_t gamma[32],
    const data_t beta[32],
    const data_t mean[32],
    const data_t inv_std[32],
    data_t out[32][50]
);

void batchnorm1d_64x25(
    const data_t in[64][25],
    const data_t gamma[64],
    const data_t beta[64],
    const data_t mean[64],
    const data_t inv_std[64],
    data_t out[64][25]
);

void batchnorm1d_128x12(
    const data_t in[128][12],
    const data_t gamma[128],
    const data_t beta[128],
    const data_t mean[128],
    const data_t inv_std[128],
    data_t out[128][12]
);

// Conv1D
void conv1d_9_32_k5_l50(
    const data_t in[9][50],
    const data_t weights[32][9][5],
    const data_t bias[32],
    data_t out[32][50]
);

void conv1d_32_64_k5_l25(
    const data_t in[32][25],
    const data_t weights[64][32][5],
    const data_t bias[64],
    data_t out[64][25]
);

void conv1d_64_128_k3_l12(
    const data_t in[64][12],
    const data_t weights[128][64][3],
    const data_t bias[128],
    data_t out[128][12]
);

// MaxPool1D
void maxpool1d_32x50_to_32x25(
    const data_t in[32][50],
    data_t out[32][25]
);

void maxpool1d_64x25_to_64x12(
    const data_t in[64][25],
    data_t out[64][12]
);

// Global Average Pooling
void global_avgpool1d_128x12_to_128(
    const data_t in[128][12],
    data_t out[128]
);

// Fully connected
void fc_128_to_6(
    const data_t in[128],
    const data_t weights[6][128],
    const data_t bias[6],
    data_t out[6]
);

// Argmax
int argmax_6(const data_t in[6]);

#endif
