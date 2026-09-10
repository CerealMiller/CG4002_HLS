#ifndef CNN_BASIC_OPS_H
#define CNN_BASIC_OPS_H

#include <ap_fixed.h>

// You can change this later
typedef float data_t;
// typedef ap_fixed<16,6> data_t;

// ReLU
data_t relu(data_t x);
void relu_1d_128(const data_t in, data_t out);
void relu_2d_32x50(const data_t in, data_t out);

// Normalization
void normalize_9(
    const data_t in,
    const data_t mean,
    const data_t inv_std,
    data_t out
);

void normalize_9x50(
    const data_t in,
    const data_t mean,
    const data_t inv_std,
    data_t out
);

// MaxPool1D
void maxpool1d_32x50_to_32x25(
    const data_t in,
    data_t out
);

void maxpool1d_64x25_to_64x12(
    const data_t in,
    data_t out
);

// Global Average Pooling
void global_avgpool1d_128x12_to_128(
    const data_t in,
    data_t out
);

// Fully connected
void fc_128_to_6(
    const data_t in,
    const data_t weights,
    const data_t bias,
    data_t out
);

// Argmax
int argmax_6(const data_t in);

#endif
