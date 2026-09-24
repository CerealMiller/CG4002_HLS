#ifndef CNN_INFERENCE_HPP
#define CNN_INFERENCE_HPP

#include "cnn_basic_ops.hpp"

void cnn_inference(
    const data_t in_flat[450],
    data_t out_logits[6],
    int &predicted_class
);

#endif