#include <iostream>
#include "./cnn_basic_ops.hpp"

using namespace std;

int not_main() {
    // --------------------------------------------
    // Test ReLU scalar
    // --------------------------------------------
    cout << "Testing relu(x):" << endl;
    cout << "relu(-3) = " << relu(-3) << endl;
    cout << "relu( 2) = " << relu( 2) << endl;
    cout << endl;

    // --------------------------------------------
    // Test normalize_9
    // --------------------------------------------
    data_t in9[9]      = {1,2,3,4,5,6,7,8,9};
    data_t mean9[9]    = {1,1,1,1,1,1,1,1,1};
    data_t invstd9[9]  = {1,1,1,1,1,1,1,1,1};
    data_t out9[9];

    normalize_9(in9, mean9, invstd9, out9);

    cout << "Testing normalize_9:" << endl;
    for (int i = 0; i < 9; i++) {
        cout << out9[i] << " ";
    }
    cout << endl << endl;

    // --------------------------------------------
    // Test maxpool1d_32x50_to_32x25
    // --------------------------------------------
    data_t mp_in[32][50];
    data_t mp_out[32][25];

    for (int c = 0; c < 32; c++) {
        for (int t = 0; t < 50; t++) {
            mp_in[c][t] = t;
        }
    }

    maxpool1d_32x50_to_32x25(mp_in, mp_out);

    cout << "Testing maxpool1d_32x50_to_32x25, channel 0:" << endl;
    for (int i = 0; i < 25; i++) {
        cout << mp_out[0][i] << " ";
    }
    cout << endl << endl;

    // --------------------------------------------
    // Test global_avgpool1d_128x12_to_128
    // --------------------------------------------
    data_t gap_in[128][12];
    data_t gap_out[128];

    for (int c = 0; c < 128; c++) {
        for (int t = 0; t < 12; t++) {
            gap_in[c][t] = 1.0;
        }
    }

    global_avgpool1d_128x12_to_128(gap_in, gap_out);

    cout << "Testing global_avgpool, first 5 outputs:" << endl;
    for (int i = 0; i < 5; i++) {
        cout << gap_out[i] << " ";
    }
    cout << endl << endl;

    // --------------------------------------------
    // Test fc_128_to_6
    // --------------------------------------------
    data_t fc_in[128];
    data_t fc_w[6][128];
    data_t fc_b[6];
    data_t fc_out[6];

    for (int j = 0; j < 128; j++) {
        fc_in[j] = 1.0;
    }

    for (int i = 0; i < 6; i++) {
        fc_b[i] = 0.0;
        for (int j = 0; j < 128; j++) {
            fc_w[i][j] = 1.0;
        }
    }

    fc_128_to_6(fc_in, fc_w, fc_b, fc_out);

    cout << "Testing fc_128_to_6:" << endl;
    for (int i = 0; i < 6; i++) {
        cout << fc_out[i] << " ";
    }
    cout << endl << endl;

    // --------------------------------------------
    // Test argmax_6
    // --------------------------------------------
    data_t logits[6] = {0.1, 0.5, -0.2, 1.7, 0.9, 0.3};
    int pred = argmax_6(logits);

    cout << "Testing argmax_6:" << endl;
    cout << "Predicted index = " << pred << endl;

    return 0;
}
