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
/*
inline Matrix leakyReLU_backward(const Matrix& grad, const Matrix& input, double alpha = 0.01) {
    Matrix output = grad;

    for (int i = 0; i < grad.rows; i++) {
        for (int j = 0; j < grad.cols; j++) {
            output.data[i][j] *= (input.data[i][j] > 0) ? 1.0 : alpha;
        }
    }
    return output;
}
*/
inline Matrix softMax(const Matrix& input) {    // Assume input is a column vector
    Matrix output = input;
    double sum_exp = 0.0;
    
    for (int i = 0; i < input.rows; i++) {
        output.data[i][0] = exp(input.data[i][0]);
        sum_exp += output.data[i][0];
    }

    for (int i = 0; i < input.rows; i++) {
        output.data[i][0] /= sum_exp;
    }

    return output;
}

#endif