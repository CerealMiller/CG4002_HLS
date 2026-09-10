// cnn_hls.hpp
#ifndef CNN_HLS_HPP
#define CNN_HLS_HPP

#include <ap_int.h>
#include <ap_fixed.h>

typedef ap_fixed<16,6> data_t;   // activations/weights
typedef ap_fixed<32,12> acc_t;   // accumulators

// Input dimensions
static const int IN_CH   = 8;
static const int IN_T    = 100;

// Layer 1
static const int C1_OUT  = 16;
static const int C1_K    = 5;
static const int C1_PAD  = 2;
static const int C1_T    = 100;   // same with pad=2, stride=1

// Pool 1
static const int P1_T    = 50;

// Layer 2
static const int C2_OUT  = 32;
static const int C2_K    = 3;
static const int C2_PAD  = 1;
static const int C2_T    = 50;    // same with pad=1, stride=1

// Pool 2
static const int P2_T    = 25;

// Output
static const int OUT_CLS = 5;

#endif
