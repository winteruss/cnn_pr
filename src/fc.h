#ifndef FC_H
#define FC_H

class FCLayer { 
  public:
    Matrix weights, grad_weights;
    Matrix bias, grad_bias;
    double learning_rate;
    Matrix input;   // Save for backpropagation

    FCLayer(int input_size, int output_size, double lr = 0.01)
    : weights(Matrix(output_size, input_size)),
      grad_weights(Matrix(output_size, input_size)),
      bias(Matrix(output_size, 1)),
      grad_bias(Matrix(output_size, 1)),
      learning_rate(lr) {
        weights.randomize();
        bias.randomize();
    }

    Matrix forward(const Matrix& input) {    // Assume input is already flattened(column vector)
        this -> input = input;
        Matrix result = weights * input + bias;
        return result;
    }

    Matrix backward(const Matrix& grad_output) {
        // Gradients w.r.t. weights, bias, and input
        grad_weights = grad_output * input.transpose();
        grad_bias = grad_output;
        Matrix grad_input = weights.transpose() * grad_output;
        return grad_input;
    }

    void update(double learning_rate) {
      weights -= grad_weights * learning_rate;
      bias -= grad_bias * learning_rate;
    }
};

#endif