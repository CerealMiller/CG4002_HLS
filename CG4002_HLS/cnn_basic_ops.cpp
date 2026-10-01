#include "./cnn_basic_ops.hpp"

// --------------------------------------------------
// ReLU
// --------------------------------------------------
data_t relu(data_t x) {
    return (x > 0) ? x : (data_t)0;
}

void relu_1d_128(const data_t in[128], data_t out[128]) {
#pragma HLS INLINE off
    for (int i = 0; i < 128; i++) {
#pragma HLS PIPELINE II=1
        out[i] = (in[i] > 0) ? in[i] : (data_t)0;
    }
}

void relu_2d_32x50(const data_t in[32][50], data_t out[32][50]) {
#pragma HLS INLINE off
    for (int c = 0; c < 32; c++) {
        for (int t = 0; t < 50; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] > 0) ? in[c][t] : (data_t)0;
        }
    }
}

void relu_2d_64x25(const data_t in[64][25], data_t out[64][25]) {
#pragma HLS INLINE off
    for (int c = 0; c < 64; c++) {
        for (int t = 0; t < 25; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] > 0) ? in[c][t] : (data_t)0;
        }
    }
}

void relu_2d_128x12(const data_t in[128][12], data_t out[128][12]) {
#pragma HLS INLINE off
    for (int c = 0; c < 128; c++) {
        for (int t = 0; t < 12; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] > 0) ? in[c][t] : (data_t)0;
        }
    }
}


// --------------------------------------------------
// Normalization
// --------------------------------------------------
void normalize_9(
    const data_t in[9],
    const data_t mean[9],
    const data_t inv_std[9],
    data_t out[9]
) {
#pragma HLS INLINE off
    for (int i = 0; i < 9; i++) {
#pragma HLS PIPELINE II=1
        out[i] = (in[i] - mean[i]) * inv_std[i];
        #pragma HLS BIND_OP variable=out op=mul impl=dsp
    }
}

void normalize_9x50(
    const data_t in[9][50],
    const data_t mean[9],
    const data_t inv_std[9],
    data_t out[9][50]
) {
#pragma HLS INLINE off
    for (int c = 0; c < 9; c++) {
        for (int t = 0; t < 50; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] - mean[c]) * inv_std[c];
            #pragma HLS BIND_OP variable=out op=mul impl=dsp
        }
    }
}

// BatchNorm Implementations
void batchnorm1d_32x50(
    const data_t in[32][50],
    const data_t gamma[32],
    const data_t beta[32],
    const data_t mean[32],
    const data_t inv_std[32],
    data_t out[32][50]
) {
#pragma HLS INLINE off
    for (int c = 0; c < 32; c++) {
        for (int t = 0; t < 50; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] - mean[c]) * inv_std[c] * gamma[c] + beta[c];
            #pragma HLS BIND_OP variable=out op=mul impl=dsp
        }
    }
}

void batchnorm1d_64x25(
    const data_t in[64][25],
    const data_t gamma[64],
    const data_t beta[64],
    const data_t mean[64],
    const data_t inv_std[64],
    data_t out[64][25]
) {
#pragma HLS INLINE off
    for (int c = 0; c < 64; c++) {
        for (int t = 0; t < 25; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] - mean[c]) * inv_std[c] * gamma[c] + beta[c];
            #pragma HLS BIND_OP variable=out op=mul impl=dsp
        }
    }
}

void batchnorm1d_128x12(
    const data_t in[128][12],
    const data_t gamma[128],
    const data_t beta[128],
    const data_t mean[128],
    const data_t inv_std[128],
    data_t out[128][12]
) {
#pragma HLS INLINE off
    for (int c = 0; c < 128; c++) {
        for (int t = 0; t < 12; t++) {
#pragma HLS PIPELINE II=1
            out[c][t] = (in[c][t] - mean[c]) * inv_std[c] * gamma[c] + beta[c];
            #pragma HLS BIND_OP variable=out op=mul impl=dsp
        }
    }
}

// --------------------------------------------------
// Conv1D: 9 -> 32, kernel=5, input length=50, padding=2
// output length=50
// --------------------------------------------------
void conv1d_9_32_k5_l50(
    const data_t in[9][50],
    const data_t weights[32][9][5],
    const data_t bias[32],
    data_t out[32][50]
) {
#pragma HLS INLINE off

    for (int oc = 0; oc < 32; oc++) {
        for (int t = 0; t < 50; t++) {
#pragma HLS PIPELINE II=1
            acc_t sum = bias[oc];

            for (int ic = 0; ic < 9; ic++) {
                for (int k = 0; k < 5; k++) {
                    int idx = t + k - 2;   // padding = 2
                    data_t x = 0;

                    if (idx >= 0 && idx < 50) {
                        x = in[ic][idx];
                    }

                    sum += weights[oc][ic][k] * x;
                    #pragma HLS BIND_OP variable=sum op=mul impl=dsp
                }
            }

            out[oc][t] = (data_t)sum;
        }
    }
}

// --------------------------------------------------
// Conv1D: 32 -> 64, kernel=5, input length=25, padding=2
// output length=25
// --------------------------------------------------
void conv1d_32_64_k5_l25(
    const data_t in[32][25],
    const data_t weights[64][32][5],
    const data_t bias[64],
    data_t out[64][25]
) {
#pragma HLS INLINE off

    for (int oc = 0; oc < 64; oc++) {
        for (int t = 0; t < 25; t++) {
#pragma HLS PIPELINE II=1
            acc_t sum = bias[oc];

            for (int ic = 0; ic < 32; ic++) {
                for (int k = 0; k < 5; k++) {
                    int idx = t + k - 2;   // padding = 2
                    data_t x = 0;

                    if (idx >= 0 && idx < 25) {
                        x = in[ic][idx];
                    }

                    sum += weights[oc][ic][k] * x;
                    #pragma HLS BIND_OP variable=sum op=mul impl=dsp
                }
            }

            out[oc][t] = (data_t)sum;
        }
    }
}

// --------------------------------------------------
// Conv1D: 64 -> 128, kernel=3, input length=12, padding=1
// output length=12
// --------------------------------------------------
void conv1d_64_128_k3_l12(
    const data_t in[64][12],
    const data_t weights[128][64][3],
    const data_t bias[128],
    data_t out[128][12]
) {
#pragma HLS INLINE off

    for (int oc = 0; oc < 128; oc++) {
        for (int t = 0; t < 12; t++) {
#pragma HLS PIPELINE II=1
            acc_t sum = bias[oc];

            for (int ic = 0; ic < 64; ic++) {
                for (int k = 0; k < 3; k++) {
                    int idx = t + k - 1;   // padding = 1
                    data_t x = 0;

                    if (idx >= 0 && idx < 12) {
                        x = in[ic][idx];
                    }

                    sum += weights[oc][ic][k] * x;
                    #pragma HLS BIND_OP variable=sum op=mul impl=dsp
                }
            }

            out[oc][t] = (data_t)sum;
        }
    }
}

// --------------------------------------------------
// MaxPool1D: 32x50 -> 32x25
// --------------------------------------------------
void maxpool1d_32x50_to_32x25(
    const data_t in[32][50],
    data_t out[32][25]
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
// --------------------------------------------------
void maxpool1d_64x25_to_64x12(
    const data_t in[64][25],
    data_t out[64][12]
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
// --------------------------------------------------
void global_avgpool1d_128x12_to_128(
    const data_t in[128][12],
    data_t out[128]
) {
#pragma HLS INLINE off
    const data_t inv12 = (data_t)0.0833333333333;

    for (int c = 0; c < 128; c++) {
#pragma HLS PIPELINE II=1
        acc_t sum = 0;
        for (int t = 0; t < 12; t++) {
            sum += in[c][t];
        }
        out[c] = (data_t)(sum * inv12);
        #pragma HLS BIND_OP variable=out op=mul impl=dsp
    }
}

// --------------------------------------------------
// Fully connected: 128 -> 6
// --------------------------------------------------
void fc_128_to_6(
    const data_t in[128],
    const data_t weights[6][128],
    const data_t bias[6],
    data_t out[6]
) {
#pragma HLS INLINE off
    for (int i = 0; i < 6; i++) {
#pragma HLS PIPELINE II=1
        acc_t sum = bias[i];
        for (int j = 0; j < 128; j++) {
            sum += weights[i][j] * in[j];
            #pragma HLS BIND_OP variable=sum op=mul impl=dsp
        }
        out[i] = (data_t)sum;
    }
}

// --------------------------------------------------
// Argmax over 6 values
// --------------------------------------------------
int argmax_6(const data_t in[6]) {
#pragma HLS INLINE off
    int max_idx = 0;
    data_t max_val = in[0]; // Fixed: changed from 'in' to 'in[0]'

    for (int i = 1; i < 6; i++) {
#pragma HLS PIPELINE II=1
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }

    return max_idx;
}
