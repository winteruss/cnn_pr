#include <iostream>

#include "util.h"
#include "training.h"
#include "model.h"
#include "dataset.h"

int main() {
    Dataset dataset;
    dataset.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_train.csv", 28, 28, 10);

    int num_conv_layers = 3;
    int fc_input_size = Model::calculate_fc_input_size(28, 28, num_conv_layers);
    int epochs = 50;

    Model model(28, 28, 10, 0.001, num_conv_layers);

    for (int i = 0; i < num_conv_layers; i++) {
        model.conv_layers[i].kernel = Matrix(3, 3);
        model.conv_layers[i].kernel.randomize(-0.1, 0.1);
        model.conv_layers[i].bias = 0.0;
    }

    model.fc.weights = Matrix(10, fc_input_size);
    model.fc.weights.randomize(-0.1, 0.1);
    model.fc.bias = Matrix(10, 1);

    trainDataset(model, dataset, epochs);
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