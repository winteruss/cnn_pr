#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "conv.h"
#include "pool.h"
#include "fc.h"
#include "activation.h"

class Model {
  public:
    std::vector<ConvLayer> conv_layers;
    std::vector<PoolLayer> pool_layers;
    FCLayer fc;

    int input_size;
    int output_size;
    double learning_rate;

    Model(int input_size, int output_size, double lr, int num_conv_layers)
      : fc(input_size, output_size, lr), input_size(input_size), output_size(output_size), learning_rate(lr) {
        for (int i = 0; i < num_conv_layers; i++) {
            conv_layers.emplace_back();
            pool_layers.emplace_back();
        }
    }

    Matrix forward(const Matrix& input) {
        Matrix x = input;
        
        for (int i = 0; i < conv_layers.size(); i++) {
            x = conv_layers[i].forward(x);
            x = ReLU(x);
            x = pool_layers[i].forward(x);
        }

        x = fc.forward(x);
        return x;
    }

    void backward(const Matrix& loss_grad) {
        return ;
    }
};

#endif