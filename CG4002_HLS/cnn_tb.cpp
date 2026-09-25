#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "cnn_inference.hpp"

#define NUM_CLASSES 6

const char* CLASS_NAMES[NUM_CLASSES] = {
    "Standing",
    "Walking (w/o crate)",
    "Bending",
    "Lifting crate",
    "Walking (w/ crate)",
    "Placing crate"
};

int main() {
    std::ifstream file_in("test_inputs.txt");
    std::ifstream file_lbl("test_labels.txt");

    if (!file_in.is_open() || !file_lbl.is_open()) {
        std::cerr << "ERROR: Could not open test dataset files!" << std::endl;
        return 1;
    }

    int confusion_matrix[NUM_CLASSES][NUM_CLASSES] = {0};
    int total_samples = 0;
    int correct_predictions = 0;

    data_t out_logits[NUM_CLASSES];
    int predicted_class = -1;
    int true_label = -1;

    std::cout << "====================================================" << std::endl;
    std::cout << "         Vitis HLS CNN Inference Testbench          " << std::endl;
    std::cout << "====================================================" << std::endl;

    while (file_lbl >> true_label) {
        hls::stream<axis_t> in_stream("in_stream");
        hls::stream<axis_t> out_stream("out_stream");

        // Read 450 values, cast to data_t, and write to input stream
        for (int i = 0; i < 450; i++) {
            float val;
            file_in >> val;
            data_t sample = (data_t)val; // Original float-to-data_t cast
            
            bool is_last = (i == 449);
            in_stream.write(data_to_axis(sample, is_last));
        }

        // Run HLS Top-Level Inference Core
        cnn_inference(in_stream, out_stream, predicted_class);

        // Read 6 output logits back from output stream
        for (int i = 0; i < NUM_CLASSES; i++) {
            axis_t out_pkt = out_stream.read();
            out_logits[i] = axis_to_data(out_pkt);
        }

        // Record Statistics
        if (true_label >= 0 && true_label < NUM_CLASSES) {
            confusion_matrix[true_label][predicted_class]++;
            if (predicted_class == true_label) {
                correct_predictions++;
            }
            total_samples++;
        }
    }

    file_in.close();
    file_lbl.close();

    if (total_samples == 0) {
        std::cerr << "ERROR: No test samples processed." << std::endl;
        return 1;
    }

    // Calculate Overall Accuracy
    float accuracy = (float)correct_predictions / total_samples * 100.0f;

    std::cout << "\nTest Samples Evaluated: " << total_samples << std::endl;
    std::cout << "Correct Predictions   : " << correct_predictions << std::endl;
    std::cout << "Overall Accuracy      : " << std::fixed << std::setprecision(2) << accuracy << "%\n" << std::endl;

    // Display Confusion Matrix
    std::cout << "--- Confusion Matrix (Rows: True, Cols: Pred) ---" << std::endl;
    std::cout << std::setw(22) << " ";
    for (int j = 0; j < NUM_CLASSES; j++) {
        std::cout << std::setw(6) << "P" << j;
    }
    std::cout << std::endl;

    for (int i = 0; i < NUM_CLASSES; i++) {
        std::cout << std::setw(20) << CLASS_NAMES[i] << " |";
        for (int j = 0; j < NUM_CLASSES; j++) {
            std::cout << std::setw(6) << confusion_matrix[i][j];
        }
        std::cout << std::endl;
    }

    // Calculate Precision & Recall Per Class
    std::cout << "\n--- Per-Class Performance Metrics ---" << std::endl;
    std::cout << std::left << std::setw(24) << "Class Name" 
              << std::setw(15) << "Precision (%)" 
              << std::setw(15) << "Recall (%)" 
              << std::setw(15) << "F1-Score (%)" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;

    for (int c = 0; c < NUM_CLASSES; c++) {
        int tp = confusion_matrix[c][c];
        int fp = 0;
        int fn = 0;

        for (int i = 0; i < NUM_CLASSES; i++) {
            if (i != c) {
                fp += confusion_matrix[i][c];
                fn += confusion_matrix[c][i];
            }
        }

        float precision = (tp + fp > 0) ? ((float)tp / (tp + fp)) * 100.0f : 0.0f;
        float recall    = (tp + fn > 0) ? ((float)tp / (tp + fn)) * 100.0f : 0.0f;
        float f1        = (precision + recall > 0) ? (2.0f * precision * recall) / (precision + recall) : 0.0f;

        std::cout << std::left << std::setw(24) << CLASS_NAMES[c]
                  << std::setw(15) << std::fixed << std::setprecision(2) << precision
                  << std::setw(15) << std::fixed << std::setprecision(2) << recall
                  << std::setw(15) << std::fixed << std::setprecision(2) << f1 << std::endl;
    }

    std::cout << "====================================================" << std::endl;

    if (accuracy > 40.0f) {
        std::cout << "TESTBENCH PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "TESTBENCH FAILED! Accuracy below target threshold." << std::endl;
        return 1;
    }
}