#ifndef CNN_INFERENCE_HPP
#define CNN_INFERENCE_HPP

#include "cnn_basic_ops.hpp"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

// 32-bit AXI-Stream Packet Structure
typedef ap_axiu<32, 0, 0, 0> axis_t;

// Helper: Pack data_t into AXI-Stream packet payload
inline axis_t data_to_axis(data_t val, bool last = false) {
    axis_t pkt;
    pkt.keep = -1;
    pkt.strb = -1;
    pkt.last = last ? 1 : 0;

#if defined(AP_FIXED_H) || defined(__AP_FIXED_H__)
    pkt.data = 0;
    pkt.data(15, 0) = val.range(15, 0);
#else
    union { data_t d; uint32_t u; } conv;
    conv.d = val;
    pkt.data = conv.u;
#endif
    return pkt;
}

// Helper: Unpack AXI-Stream packet payload into data_t
inline data_t axis_to_data(const axis_t &pkt) {
    data_t val;
#if defined(AP_FIXED_H) || defined(__AP_FIXED_H__)
    val.range(15, 0) = pkt.data(15, 0);
#else
    union { data_t d; uint32_t u; } conv;
    conv.u = (uint32_t)pkt.data;
    val = conv.d;
#endif
    return val;
}

// Top-level HLS interface signature
void cnn_inference(
    hls::stream<axis_t> &in_stream,
    hls::stream<axis_t> &out_stream,
    int &predicted_class
);

#endif