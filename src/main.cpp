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