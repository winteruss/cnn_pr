#ifndef FC_H
#define FC_H

class FCLayer { 
  private:
    Matrix last_input;

  public:
    Matrix weights;
    Matrix bias;
    double learning_rate;

    FCLayer(int input_size, int output_size, double lr = 0.01)
    : weights(Matrix(output_size, input_size)), bias(Matrix(output_size, 1)), learning_rate(lr) {
        weights.randomize();
        bias.randomize();
    }

    Matrix forward(const Matrix& input) {
        last_input = input;
        Matrix result = weights * input + bias;
        return result;
    }

    Matrix backward(const Matrix& d_output) {
        Matrix d_weights = d_output * last_input.T();
        Matrix d_bias = d_output;

        weights -= d_weights * learning_rate;
        bias -= d_bias * learning_rate;

        return weights.T() * d_output;
    }
};

#endif