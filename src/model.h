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

    std::vector<Matrix> pre_activation;    // Store pre-activation values for backpropagation

    Model(int input_size, int output_size, double lr, int num_conv_layers)
    // Adjust input size of FC layer regarding pooling layers, assuming 2x2 and stride 2
    : fc(input_size / (1 << num_conv_layers), output_size, lr), input_size(input_size), output_size(output_size), learning_rate(lr) {  
        for (int i = 0; i < num_conv_layers; i++) {
            conv_layers.emplace_back();
            pool_layers.emplace_back();
        }
        pre_activation.resize(num_conv_layers);
    }

    std::pair<Matrix, double> forward(const Matrix& input, const Matrix& target) {
        Matrix x = input;
        
        for (int i = 0; i < conv_layers.size(); i++) {
            x = conv_layers[i].forward(x);
            pre_activation[i] = x;
            x = leakyReLU(x);
            x = pool_layers[i].forward(x);
        }

        x = x.flatten();
        x = fc.forward(x);
        
        double loss = crossEntropyLoss(softmax(x), target);
        return {x, loss};
    }

    void backward(const Matrix& loss_grad) {
        Matrix grad = fc.backward(loss_grad);

        for (int i = conv_layers.size() - 1; i >= 0; i--) {
            grad = pool_layers[i].backward(grad);
            grad = leakyReLU_backward(grad, pre_activation[i]);
            grad = conv_layers[i].backward(grad);
        }
    }
};

#endif