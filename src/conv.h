#ifndef CONV_H
#define CONV_H

class ConvLayer {
  public:
    Matrix kernel;

    ConvLayer(const Matrix& k) : kernel(k) {}

    Matrix forward(const Matrix& input) const {
        int pad_top = (kernel.rows - 1) / 2;
        int pad_left = (kernel.cols - 1) / 2;
        int pad_bottom = pad_top + (kernel.rows % 2 == 0 ? 1 : 0);
        int pad_right = pad_left + (kernel.cols % 2 == 0 ? 1 : 0);

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
};

#endif