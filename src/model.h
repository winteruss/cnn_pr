#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "conv.h"
#include "pool.h"
#include "fc.h"
#include "activation.h"
#include "loss.h"

class Model {
  public:
    std::vector<ConvLayer> conv_layers;
    std::vector<PoolLayer> pool_layers;
    FCLayer fc;

    int input_size;
    int output_size;
    double learning_rate;

    // std::vector<Matrix> pre_activation;    // Store pre-activation values for backpropagation

    Model(int input_rows, int input_cols, int output_size, double lr, int num_conv_layers)
    // Adjust input size of FC layer regarding pooling layers, assuming 2x2 and stride 2
    : fc(calculate_fc_input_size(input_rows, input_cols, num_conv_layers), output_size, lr), output_size(output_size), learning_rate(lr) {
        int rows = input_rows;
        int cols = input_cols;
        for (int i = 0; i < num_conv_layers; i++) {
            conv_layers.emplace_back();
            pool_layers.emplace_back();
            rows = (rows - 2) / 2 + 1;
            cols = (cols - 2) / 2 + 1;
        }
        fc = FCLayer(rows * cols, output_size, lr);
        // pre_activation.resize(num_conv_layers);
    }

    std::pair<Matrix, double> forward(const Matrix& input, const Matrix& target) {
        Matrix x = input;
        
        for (int i = 0; i < conv_layers.size(); i++) {
            x = conv_layers[i].forward(x);
            // pre_activation[i] = x;
            x = leakyReLU(x);
            x = pool_layers[i].forward(x);
        }

        x = x.flatten();
        x = fc.forward(x);
        
        double loss = crossEntropyLoss(softMax(x), target);
        return {x, loss};
    }
/*
    void backward(const Matrix& loss_grad) {
        Matrix grad = fc.backward(loss_grad);

        for (int i = conv_layers.size() - 1; i >= 0; i--) {
            grad = pool_layers[i].backward(grad);
            grad = leakyReLU_backward(grad, pre_activation[i]);
            grad = conv_layers[i].backward(grad);
        }
    }
*/

  private:
    static int calculate_fc_input_size(int rows, int cols, int num_conv_layers) {
        int reduced_rows = rows;
        int reduced_cols = cols;
        for (int i = 0; i < num_conv_layers; i++) {
            reduced_rows = (reduced_rows - 2) / 2 + 1;
            reduced_cols = (reduced_cols - 2) / 2 + 1;
        }
        return reduced_rows * reduced_cols;
    }
};

#endif