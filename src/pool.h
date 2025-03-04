#ifndef POOL_H
#define POOL_H

#include "util.h"

class PoolLayer {
  private:
    int pool_r, pool_c;
    int stride_r, stride_c;
    Matrix input;   // Save for backpropagation
    Matrix max_mask;    // Store Max Positions

  public:
    PoolLayer(int pr = 2, int pc = 2, int sr = 2, int sc = 2)   // Default: 2x2 pooling with stride 2
    : pool_r(pr), pool_c(pc), stride_r(sr), stride_c(sc) {}

    Matrix forward(const Matrix& input) {
        this -> input = input;
        int output_rows = (input.rows - pool_r) / stride_r + 1;
        int output_cols = (input.cols - pool_c) / stride_c + 1;
        Matrix pooled(output_rows, output_cols);
        max_mask = Matrix(input.rows, input.cols);

        for (int i = 0; i < output_rows; i++) {
            for (int j = 0; j < output_cols; j++) {
                double max_val = -1e9;
                int max_row = -1, max_col = -1;
                for (int m = 0; m < pool_r; m++) {
                    for (int n = 0; n < pool_c; n++) {
                        int row = i * stride_r + m;
                        int col = j * stride_c + n;
                        if (input.data[row][col] > max_val) {
                            max_row = row;
                            max_col = col;
                            max_val = max(max_val, input.data[row][col]);
                        }
                    }
                }
                pooled.data[i][j] = max_val;
                max_mask.data[max_row][max_col] = 1;
            }
        }
        return pooled;
    }

    Matrix backward(const Matrix& grad_output) {
        Matrix grad_input(input.rows, input.cols);
        for (int i = 0; i < grad_output.rows; i++) {
            for (int j = 0; j < grad_output.cols; j++) {
                for (int m = 0; m < pool_r; m++) {
                    for (int n = 0; n < pool_c; n++) {
                        int row = i * stride_r + m;
                        int col = j * stride_c + n;
                        if (row < input.rows && col < input.cols && max_mask.data[row][col] == 1) {
                            grad_input.data[row][col] = grad_output.data[i][j];
                        }
                    }
                }
            }
        }
        return grad_input;
    }

};

#endif