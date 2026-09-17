#ifndef CNN_BASIC_OPS_H
#define CNN_BASIC_OPS_H

#include <ap_fixed.h>

typedef float data_t;

// ReLU
data_t relu(data_t x);
void relu_1d_128(const data_t in[128], data_t out[128]);
void relu_2d_32x50(const data_t in[32][50], data_t out[32][50]);

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
