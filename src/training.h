#ifndef TRAIN_H
#define TRAIN_H

#include "model.h"
/*
void train(Model& model, const std::vector<std::pair<Matrix, Matrix>>& dataset, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << std::endl;

        for (int i = 0; i < dataset.size(); i++) {
            const auto& [input, target] = dataset[i];

            auto [output, loss] = model.forward(input, target);
            Matrix loss_grad = softMax(output) - target;  // Derivative of Cross Entropy Loss
            model.backward(loss_grad);

            std::cout << "  Sample " << i + 1 << " - Loss: " << loss << std::endl;
        }
    }
}*/

void train(Model& model, const Matrix& input, const Matrix& target, int epochs) {
    Matrix normalized_input = input.normalize();
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << std::endl;

        auto [logits, loss] = model.forward(normalized_input, target);
        model.backward(logits, target);
        model.update();

        std::cout << "  Loss: " << loss << std::endl;
    }

}

#endif
