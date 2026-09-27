#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "cnn_inference.hpp"

using namespace std;

// Match top module packet type definition
typedef ap_axiu<32, 0, 0, 0> axis_t;

#define INPUT_MAX_WORDS 450
#define OUTPUT_WORDS    6

int main() {
    int errors = 0;

    // AXI Streams for DUT
    hls::stream<axis_t> in_stream("in_stream");
    hls::stream<axis_t> out_stream("out_stream");

    // File handles for test vectors
    ifstream input_file("test_inputs.txt");
    ifstream label_file("test_labels.txt");

    bool use_file_inputs = input_file.is_open() && label_file.is_open();

    if (!use_file_inputs) {
        cout << "[WARNING] Could not open 'test_inputs.txt' or 'test_labels.txt'." << endl;
        cout << "[INFO] Running single synthetic sample test case..." << endl;
    } else {
        cout << "[INFO] Successfully loaded test vector files." << endl;
    }

    cout << "==================================================" << endl;
    cout << " Starting HLS AXI-Stream CNN Inference Testbench  " << endl;
    cout << "==================================================" << endl;

    int test_case_cnt = 0;
    int correct_predictions = 0;

    // Run until EOF or exactly 1 loop if using synthetic data
    while (true) {
        int expected_label = 0;
        int raw_sample_hex = 0;

        if (use_file_inputs) {
            if (!(label_file >> expected_label)) break; // EOF reached
        } else {
            if (test_case_cnt >= 1) break; // End single synthetic run
            expected_label = 3; // Arbitrary target label for synthetic test
        }

        test_case_cnt++;

        // -------------------------------------------------
        // 1. Pack 450 samples into Input AXI-Stream
        // -------------------------------------------------
        for (int i = 0; i < INPUT_MAX_WORDS; i++) {
            axis_t in_pkt;

            if (use_file_inputs) {
                input_file >> hex >> raw_sample_hex;
                in_pkt.data = (ap_int<16>)raw_sample_hex;
            } else {
                // Generate deterministic synthetic data
                in_pkt.data = (ap_int<16>)(i % 100);
            }

            in_pkt.keep = -1; // 0xF: All byte lanes valid
            in_pkt.strb = -1;
            in_pkt.last = (i == INPUT_MAX_WORDS - 1) ? 1 : 0; // Assert TLAST on 450th word

            in_stream.write(in_pkt);
        }

        // -------------------------------------------------
        // 2. Execute Top-Level HLS Design Under Test (DUT)
        // -------------------------------------------------
        cnn_inference(in_stream, out_stream);

        // -------------------------------------------------
        // 3. Unpack 6 Output Logits & Verify TLAST
        // -------------------------------------------------
        data_t received_logits[OUTPUT_WORDS];
        data_t max_logit = -9999.0;
        int predicted_class = 0;

        for (int i = 0; i < OUTPUT_WORDS; i++) {
            if (out_stream.empty()) {
                cout << "[ERROR] Test Case " << test_case_cnt 
                     << ": out_stream underflow at word " << i << endl;
                return 1;
            }

            axis_t out_pkt = out_stream.read();
            
            // Extract logit value from low 16-bits
            ap_int<16> raw_logit = (ap_int<16>)out_pkt.data(15, 0);
            received_logits[i] = (data_t)raw_logit;

            // Argmax Tracking
            if (i == 0 || received_logits[i] > max_logit) {
                max_logit = received_logits[i];
                predicted_class = i;
            }

            // Verify AXI-Stream TLAST assertion timing
            bool expected_last = (i == OUTPUT_WORDS - 1);
            if (out_pkt.last != expected_last) {
                cout << "[ERROR] TLAST mismatch at index " << i 
                     << " (Got: " << out_pkt.last << ", Expected: " << expected_last << ")" << endl;
                errors++;
            }
        }

        // -------------------------------------------------
        // 4. Accuracy Evaluation
        // -------------------------------------------------
        if (predicted_class == expected_label) {
            correct_predictions++;
            cout << "[PASS] Test Case " << setfill(' ') << setw(3) << test_case_cnt 
                 << " | Predicted: " << predicted_class 
                 << " | Expected: " << expected_label << endl;
        } else {
            cout << "[FAIL] Test Case " << setfill(' ') << setw(3) << test_case_cnt 
                 << " | Predicted: " << predicted_class 
                 << " | Expected: " << expected_label << endl;
        }
    }

    if (use_file_inputs) {
        input_file.close();
        label_file.close();
    }

    // -------------------------------------------------
    // Final Summary & Return Status for Vitis C-Sim
    // -------------------------------------------------
    cout << "==================================================" << endl;
    cout << " C Simulation Complete" << endl;
    cout << " Total Test Cases : " << test_case_cnt << endl;
    cout << " Correct Hits     : " << correct_predictions << endl;
    if (test_case_cnt > 0) {
        double accuracy = (double)correct_predictions * 100.0 / test_case_cnt;
        cout << " Accuracy         : " << fixed << setprecision(2) << accuracy << "%" << endl;
    }
    cout << "==================================================" << endl;

    // Vitis HLS C-Simulation requires returning 0 on success, non-zero on failure
    return (errors > 0) ? 1 : 0;
}