#ifndef CONV_H
#define CONV_H

class ConvLayer {
  private:
    Matrix last_input;
    int pad_top, pad_left, pad_bottom, pad_right;

  public:
    Matrix kernel;

    ConvLayer() : kernel(3, 3) {
        kernel.randomize();
        calculate_padding();
    }

    ConvLayer(const Matrix& k) : kernel(k) {
        calculate_padding();
    }

    void calculate_padding() {
        pad_top = (kernel.rows - 1) / 2;
        pad_left = (kernel.cols - 1) / 2;
        pad_bottom = pad_top + (kernel.rows % 2 == 0 ? 1 : 0);
        pad_right = pad_left + (kernel.cols % 2 == 0 ? 1 : 0);
    }

    Matrix forward(const Matrix& input) {
        last_input = input;

        Matrix padded_input = input.pad(pad_top, pad_left, pad_bottom, pad_right);
        Matrix output(input.rows, input.cols);

        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < kernel.rows; ki++) {
                    for (int kj = 0; kj < kernel.cols; kj++) {
                        sum += padded_input.data[i+ki][j+kj] * kernel.data[ki][kj];
                    }
                }
                output.data[i][j] = sum;
            }
        }
        return output;
    }

    Matrix backward(const Matrix& d_out) {
        Matrix d_kernel(kernel.rows, kernel.cols);
        for (int i = 0; i < d_kernel.rows; i++) {
            for (int j = 0; j < d_kernel.cols; j++) {
                double sum = 0.0;
                for (int r = 0; r < d_out.rows; r++) {
                    for (int c = 0; c < d_out.cols; c++) {
                        sum += d_out.data[r][c] * last_input.data[r+i][c+j];
                    }
                }
                d_kernel.data[i][j] = sum;
            }
        }
        
        Matrix d_input(last_input.rows + pad_top + pad_bottom, last_input.cols + pad_left + pad_right);
        Matrix flipped_kernel = kernel.flip();

        for (int i = 0; i < d_out.rows; i++) {
            for (int j = 0; j < d_out.cols; j++) {
                for (int ki = 0; ki < kernel.rows; ki++) {
                    for (int kj = 0; kj < kernel.cols; kj++) {
                        d_input.data[i+ki][j+kj] += d_out.data[i][j] * flipped_kernel.data[ki][kj];
                    }
                }
            }
        }
        return d_input.slice(pad_top, pad_top + last_input.rows, pad_left, pad_left + last_input.cols);
    }
};

#endif