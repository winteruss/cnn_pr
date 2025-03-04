#ifndef CONV_H
#define CONV_H

class ConvLayer {
  public:
    Matrix kernel, grad_kernel;
    double bias, grad_bias;
    Matrix input;   // Save for backpropagation

    ConvLayer() : kernel(3, 3), grad_kernel(3, 3), bias(rand() / (double)RAND_MAX), grad_bias(0.0) {
        kernel.randomize();
    }

    ConvLayer(const Matrix& k) : kernel(k) {}

    Matrix forward(const Matrix& input) {
        this -> input = input;
        Matrix padded_input = input.pad(kernel.cols / 2);   // Suppose that kernel is square
        Matrix output(input.rows, input.cols);

        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < kernel.rows; ki++) {
                    for (int kj = 0; kj < kernel.cols; kj++) {
                        sum += padded_input.data[i+ki][j+kj] * kernel.data[ki][kj];
                    }
                }
                output.data[i][j] = sum + bias;
            }
        }
        return output;
    }

    Matrix backward(const Matrix& grad_out) {
        Matrix padded_input = input.pad(kernel.cols / 2);
        Matrix grad_input(input.rows, input.cols);
        grad_kernel = Matrix(kernel.rows, kernel.cols);
        grad_bias = 0.0;

        // Gradients w.r.t. kernel and bias
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                for (int ki = 0; ki < kernel.rows; ki++) {
                    for (int kj = 0; kj < kernel.cols; kj++) {
                        grad_kernel.data[ki][kj] += grad_out.data[i][j] * padded_input.data[i+ki][j+kj];
                    }
                }
                grad_bias += grad_out.data[i][j];
            }
        }

        // Gradients w.r.t. input
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < kernel.rows; ki++) {
                    for (int kj = 0; kj < kernel.cols; kj++) {
                        int oi = i - ki + kernel.rows / 2;
                        int oj = i - kj + kernel.cols / 2;
                        if (oi >= 0 && oi < grad_out.rows && oj >= 0 && oj < grad_out.cols) {
                            sum += grad_out.data[oi][oj] * kernel.data[ki][kj];
                        }
                    }
                }
                grad_input.data[i][j] = sum;
            }
        }
        return grad_input;
    }

    void update(double lr) {
        kernel = kernel - grad_kernel * lr;
        bias -= lr * grad_bias;
    }
};

#endif