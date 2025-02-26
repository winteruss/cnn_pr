#ifndef FC
#define FC

class FCLayer {
  public:
    Matrix weights;
    Matrix bias;
    double learning_rate;

    FCLayer(int input_size, int output_size, double lr = 0.01) : weights(Matrix(output_size, input_size)), bias(Matrix(output_size, 1)), learning_rate(lr) {
        weights.randomize();
        bias.randomize();
    }

    Matrix forward(const Matrix& input) const {
        return weights * input + bias;
    }

    Matrix backward(const Matrix& input, const Matrix& d_output) {
        Matrix d_weights = d_output * input.T();
        Matrix d_bias = d_output;

        weights -= d_weights * learning_rate;
        bias -= d_bias * learning_rate;

        return weights.T() * d_output;
    }
    
};

#endif