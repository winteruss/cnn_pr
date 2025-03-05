#include <iostream>

#include "util.h"
#include "training.h"
#include "model.h"
#include "dataset.h"

int main() {
    Dataset dataset;
    dataset.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_train.csv", 28, 28, 10);
    Model model(28, 28, 10, 0.001, 2);

    model.conv_layers[0].kernel = Matrix(3, 3, 1);
    model.conv_layers[0].bias = 0.0;
    model.conv_layers[1].kernel = Matrix(3, 3, 0.9);
    model.conv_layers[1].bias = 0.0;

    model.fc.weights = Matrix(10, 49, 0.1);
    model.fc.bias = Matrix(10, 1, 0.0);

    trainDataset(model, dataset, 50);
    model.save("trained_model.txt");

    std::cout << "\nFinal Predictions:\n";
    for (size_t i = 0; i < dataset.size(); i++) {
        const auto& [input, target] = dataset[i];
        Matrix norm_input = input.normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        std::cout << "Image " << i + 1 << " Loss: " << loss << "\nPredictions:\n";
        probs.print();
    }

    std::cout << "\nGuesses:\n";
    for (size_t i = 0; i < dataset.size(); i++) {
        const auto& [input, target] = dataset[i];
        Matrix norm_input = input.normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        int guess = argmax(probs);
        int label = argmax(target);
        std::cout << "Image " << i + 1 << " | Guess: " << guess << " Label: " << label << "\n";
    }

    int correct = 0;
    for (size_t i = 0; i < dataset.size(); i++) {
        const auto& [input, target] = dataset[i];
        Matrix norm_input = input.normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        int guess = argmax(probs);
        int label = argmax(target);
        if (guess == label) correct++;
    }
    std::cout << "\nAccuracy: " << correct << "/" << dataset.size() << " (" << (static_cast<double>(correct) / dataset.size()) * 100 << "%)\n";

    return 0;
}