#include "./cnn_basic_ops.hpp"

// --------------------------------------------------
// ReLU
// --------------------------------------------------
data_t relu(data_t x) {
    return (x > 0) ? x : data_t(0);
}

void relu_1d_128(const data_t in, data_t out) {
#pragma HLS INLINE off
    for (int i = 0; i < 128; i++) {
#pragma HLS PIPELINE II=1
        out[i] = (in[i] > 0) ? in[i] : data_t(0);
    }
}

void relu_2d_32x50(const data_t in, data_t out) {
#pragma HLS INLINE off
    for (int c = 0; c < 32; c++) {
        for (int t = 0; t < 50; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] > 0) ? in[c][t] : data_t(0);
        }
    }
}

// --------------------------------------------------
// Normalization
// out[i] = (in[i] - mean[i]) * inv_std[i]
// --------------------------------------------------
void normalize_9(
    const data_t in,
    const data_t mean,
    const data_t inv_std,
    data_t out
) {
#pragma HLS INLINE off
    for (int i = 0; i < 9; i++) {
#pragma HLS PIPELINE II=1
        out[i] = (in[i] - mean[i]) * inv_std[i];
    }
}

void normalize_9x50(
    const data_t in,
    const data_t mean,
    const data_t inv_std,
    data_t out
) {
#pragma HLS INLINE off
    for (int c = 0; c < 9; c++) {
        for (int t = 0; t < 50; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] - mean[c]) * inv_std[c];
        }
    }
}

// --------------------------------------------------
// MaxPool1D: 32x50 -> 32x25
// kernel=2, stride=2
// out[c][i] = max(in[c][2i], in[c][2i+1])
// --------------------------------------------------
void maxpool1d_32x50_to_32x25(
    const data_t in,
    data_t out
) {
#pragma HLS INLINE off
    for (int c = 0; c < 32; c++) {
        for (int i = 0; i < 25; i++) {
#pragma HLS PIPELINE II=1
            data_t a = in[c][2 * i];
            data_t b = in[c][2 * i + 1];
            out[c][i] = (a > b) ? a : b;
        }
    }
}

// --------------------------------------------------
// MaxPool1D: 64x25 -> 64x12
// kernel=2, stride=2
// Uses first 24 values, ignores last if odd length
// --------------------------------------------------
void maxpool1d_64x25_to_64x12(
    const data_t in,
    data_t out
) {
#pragma HLS INLINE off
    for (int c = 0; c < 64; c++) {
        for (int i = 0; i < 12; i++) {
#pragma HLS PIPELINE II=1
            data_t a = in[c][2 * i];
            data_t b = in[c][2 * i + 1];
            out[c][i] = (a > b) ? a : b;
        }
    }
}

// --------------------------------------------------
// Global Average Pooling: 128x12 -> 128
// out[c] = sum(in[c][0..11]) / 12
// --------------------------------------------------
void global_avgpool1d_128x12_to_128(
    const data_t in,
    data_t out
) {
#pragma HLS INLINE off
    const data_t inv12 = (data_t)0.0833333333333;

    for (int c = 0; c < 128; c++) {
#pragma HLS PIPELINE II=1
        data_t sum = 0;
        for (int t = 0; t < 12; t++) {
            sum += in[c][t];
        }
        out[c] = sum * inv12;
    }
}

// --------------------------------------------------
// Fully connected: 128 -> 6
// out[i] = bias[i] + sum_j weights[i][j] * in[j]
// --------------------------------------------------
void fc_128_to_6(
    const data_t in,
    const data_t weights,
    const data_t bias,
    data_t out
) {
#pragma HLS INLINE off
    for (int i = 0; i < 6; i++) {
#pragma HLS PIPELINE II=1
        data_t sum = bias[i];
        for (int j = 0; j < 128; j++) {
            sum += weights[i][j] * in[j];
        }
        out[i] = sum;
    }
}

// --------------------------------------------------
// Argmax over 6 values
// --------------------------------------------------
int argmax_6(const data_t in) {
#pragma HLS INLINE off
    int max_idx = 0;
    data_t max_val = in;

    for (int i = 1; i < 6; i++) {
#pragma HLS PIPELINE II=1
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }

    return max_idx;
}
