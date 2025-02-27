#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

#include "util.h"

inline Matrix leakyReLU(const Matrix& input, double alpha = 0.01) {
    Matrix output = input;

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            output.data[i][j] = (input.data[i][j] > 0) ? input.data[i][j] : alpha * input.data[i][j];
        }
    }
    return output;
}

inline Matrix leakyReLU_backward(const Matrix& grad, const Matrix& input, double alpha = 0.01) {
    Matrix output = grad;

    for (int i = 0; i < grad.rows; i++) {
        for (int j = 0; j < grad.cols; j++) {
            output.data[i][j] *= (input.data[i][j] > 0) ? 1.0 : alpha;
        }
    }
    return output;
}

inline Matrix softmax(const Matrix& input) {
    Matrix output = input;
    for (int i = 0; i < input.rows; i++) {
        double sum_exp = 0.0;

        for (int j = 0; j < input.cols; j++) {
            output.data[i][j] = exp(input.data[i][j]);
            sum_exp += output.data[i][j];
        }

        for (int j = 0; j < input.cols; j++) {
            output.data[i][j] /= sum_exp;
        }
    }
    return output;
}

#endif