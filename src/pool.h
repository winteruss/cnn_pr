#ifndef POOL_H
#define POOL_H

#include "util.h"

class PoolLayer {
  private:
    int pool_r, pool_c;
    int stride_r, stride_c;

  public:
    PoolLayer(int pr = 2, int pc = 2, int sr = 2, int sc = 2) : pool_r(pr), pool_c(pc), stride_r(sr), stride_c(sc) {}

    Matrix forward(const Matrix& input) const {
        int output_rows = (input.rows - pool_r) / stride_r + 1;
        int output_cols = (input.cols - pool_c) / stride_c + 1;
        Matrix pooled(output_rows, output_cols);

        for (int i = 0; i < output_rows; i++) {
            for (int j = 0; j < output_cols; j++) {
                double max_val = -1e9;
                for (int m = 0; m < pool_r; m++) {
                    for (int n = 0; n < pool_c; n++) {
                        max_val = max(max_val, input.data[i * stride_r + m][j * stride_c + n]);
                    }
                }
                pooled.data[i][j] = max_val;
            }
        }
        return pooled;
    }
};

#endif