#include <iostream>

#include "util.h"
#include "loadCSV.h"
#include "matrix.h"
#include "conv.h"
#include "pool.h"
#include "fc.h"
#include "activation.h"
#include "loss.h"
#include "training.h"
#include "model.h"
/*
int main() {
    std::vector<Matrix> images, labels;
    DataLoader::loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_train.csv", images, labels);
    Model model(784, 10, 0.01, 2);
    int epochs = 10;
    int batch_size = 5;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (size_t i = 0; i < images.size(); i++) {
            Matrix input = images[i].flatten();
            Matrix target = labels[i];

            auto [output, loss] = model.forward(input, target);
            total_loss += loss;

            Matrix loss_grad = softmax(output) - target;
            model.backward(loss_grad);
        }
        std::cout << "Epoch " << epoch + 1 << " Loss: " << total_loss / images.size() << std::endl;
    }

    return 0;
}
*/

int main() {
    Matrix input = {{
        { 1,  2,  3,  4,  5},
        { 6,  7,  8,  9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    }};

    Matrix target = {{
        {0},
        {1},
        {0}
    }};

    Model model(5, 5, 3, 0.01, 2);
    model.conv_layers[0].kernel = Matrix({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    model.conv_layers[0].bias = 0.0;
    model.conv_layers[1].kernel = Matrix({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    model.conv_layers[1].bias = 0.0;
    model.fc.weights = Matrix({{0.1}, {0.5}, {0.9}});
    model.fc.bias = Matrix({{0.1}, {0.2}, {0.3}});

    train(model, input, target, 5);
    model.save("trained_model.txt");

    return 0;
}