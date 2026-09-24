#include "cnn_inference.hpp"
#include "cnn_params.hpp"

void cnn_inference(
    const data_t in_flat[450],
    data_t out_logits[6],
    int &predicted_class
) {
    // 1. Generate standalone ap_start, ap_done, ap_idle, ap_ready wires 
    // for your Verilog FSM to drive/monitor directly:
    #pragma HLS INTERFACE mode=ap_ctrl_hs port=return

    // 2. Generate simple data buses with valid handshakes for array I/O
    #pragma HLS INTERFACE mode=bram port=in_flat
    #pragma HLS INTERFACE mode=bram port=out_logits
    #pragma HLS INTERFACE mode=ap_vld port=predicted_class

    // Intermediate Buffers
    data_t input_2d[9][50];
    data_t input_norm[9][50];

    data_t conv1_out[32][50];
    data_t bn1_out[32][50];
    data_t relu1_out[32][50];
    data_t pool1_out[32][25];

    data_t conv2_out[64][25];
    data_t bn2_out[64][25];
    data_t relu2_out[64][25];
    data_t pool2_out[64][12];

    data_t conv3_out[128][12];
    data_t bn3_out[128][12];
    data_t relu3_out[128][12];

    data_t gap_out[128];

    // Reshape Flat Input (450) -> [9][50] (channel-major required by Conv1D)
    for (int t = 0; t < 50; t++) {
        for (int c = 0; c < 9; c++) {
#pragma HLS PIPELINE II=1
            input_2d[c][t] = in_flat[t * 9 + c];
        }
    }

    // 1. Input Normalization
    normalize_9x50(input_2d, input_mean, input_inv_std, input_norm);

    // 2. Conv1 -> BN1 -> ReLU1 -> Pool1
    conv1d_9_32_k5_l50(input_norm, conv1_weight, conv1_bias, conv1_out);
    batchnorm1d_32x50(conv1_out, bn1_gamma, bn1_beta, bn1_mean, bn1_inv_std, bn1_out);
    relu_2d_32x50(bn1_out, relu1_out);
    maxpool1d_32x50_to_32x25(relu1_out, pool1_out);

    // 3. Conv2 -> BN2 -> ReLU2 -> Pool2
    conv1d_32_64_k5_l25(pool1_out, conv2_weight, conv2_bias, conv2_out);
    batchnorm1d_64x25(conv2_out, bn2_gamma, bn2_beta, bn2_mean, bn2_inv_std, bn2_out);
    relu_2d_64x25(bn2_out, relu2_out);
    maxpool1d_64x25_to_64x12(relu2_out, pool2_out);

    // 4. Conv3 -> BN3 -> ReLU3
    conv1d_64_128_k3_l12(pool2_out, conv3_weight, conv3_bias, conv3_out);
    batchnorm1d_128x12(conv3_out, bn3_gamma, bn3_beta, bn3_mean, bn3_inv_std, bn3_out);
    relu_2d_128x12(bn3_out, relu3_out);

    // 5. Global Avg Pool -> Fully Connected -> Argmax
    global_avgpool1d_128x12_to_128(relu3_out, gap_out);
    fc_128_to_6(gap_out, fc_weight, fc_bias, out_logits);
    predicted_class = argmax_6(out_logits);
}