#ifndef TRAIN_H
#define TRAIN_H

#include <iostream>

#include "matrix.h"
#include "dataset.h"
#include "model.h"

void trainImage(Model& model, const Matrix& input, const Matrix& target, int epochs) {
    Matrix normalized_input = input.normalize();
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << std::endl;

        auto [logits, loss] = model.forward(normalized_input, target);
        model.backward(logits, target);
        model.update();

        std::cout << "  Loss: " << loss << std::endl;
    }

}

void trainDataset(Model& model, const Dataset& dataset, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << std::endl;
        double total_loss = 0.0;

        for (int i = 0; i < dataset.size(); i++) {
            const auto& [input, target] = dataset[i];
            Matrix normalized_input = input.normalize();
            auto [logits, loss] = model.forward(normalized_input, target);
            model.backward(logits, target);
            model.update();
            total_loss += loss;
        }

        double avg_loss = total_loss / dataset.size();
        std::cout << "  Avg Loss: " << avg_loss << std::endl;
    }
}

#endif
