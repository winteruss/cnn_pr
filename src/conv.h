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
        Matrix output = padded_input.correlate(kernel) + bias;
        return output;
    }

    Matrix backward(const Matrix& grad_out) {
        Matrix padded_input = input.pad(kernel.cols / 2);
        Matrix grad_input(input.rows, input.cols);
        grad_kernel = Matrix(kernel.rows, kernel.cols);
        grad_bias = 0.0;

        // Gradients w.r.t. kernel, bias and input
        grad_kernel = padded_input.correlate(grad_out);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                grad_bias += grad_out.data[i][j];
            }
        }
        grad_input = grad_out.correlate(kernel, true);
        return grad_input;
    }

    void update(double lr) {
        kernel = kernel - grad_kernel * lr;
        bias -= lr * grad_bias;
    }
};

#endif