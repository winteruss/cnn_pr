#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

#include "util.h"

inline Matrix ReLU(const Matrix& input) {
    Matrix output = input;

    for (auto& row : output.data) {
        for (double& val : row) {
            val = max(val, 0.0);
        }
    }
    return output;
}

inline Matrix Softmax(const Matrix& input) {
    Matrix output = input;
    double sum_exp = 0.0;

    for (auto& row : output.data) {
        for (double& val : row) {
            val = exp(val);
            sum_exp += val;
        }
    }

    for (auto& row : output.data) {
        for (double& val : row) {
            val /= sum_exp;
        }
    }

    return output;
}

#endif