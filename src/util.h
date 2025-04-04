#ifndef UTIL_H
#define UTIL_H

#include "matrix.h"

inline int argmax(const Matrix& mat) {
    double max_val = mat.data[0][0];
    int max_idx = 0;
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (mat.data[i][j] > max_val) {
                max_val = mat.data[i][j];
                max_idx = i;
            }
        }
    }
    return max_idx;
}

#endif