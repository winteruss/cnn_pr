#include <iostream>

#include "util.h"
#include "matrix.h"
#include "conv.h"
#include "pool.h"
#include "fc.h"
#include "activation.h"
#include "loss.h"
#include "training.h"
#include "model.h"

int test() {
    Matrix image({
        {0, 2, 1, 4, 3},
        {4, 0, 1, 5, 2},
        {3, 3, 4, 1, 7},
        {3, 4, 4, 5, 4},
        {1, 2, 1, 1, 0}
    });

    Matrix kernel({
        {1, 0, 1},
        {1, 1, 1},
        {1, 0, 1}
    });

    Matrix ground_truth(3, 1);
    ground_truth.data[0][0] = 1;  

    ConvLayer conv(kernel);
    Matrix conv_out = conv.forward(image);
    conv_out.print();

    PoolLayer pool;
    Matrix pool_out = pool.forward(image);
    pool_out.print();

    Matrix flattened_input = pool_out.flatten();

    FCLayer fc(flattened_input.rows, 3);
    Matrix fc_out = fc.forward(flattened_input);
    fc_out.print();
    Matrix softmax_out = Softmax(fc_out);
    softmax_out.print();
    double loss = CrossEntropyLoss(softmax_out, ground_truth);
    std::cout << "Loss: " << loss << std::endl;

    return 0;
}

/*int main() {
    std::vector<Matrix> inputs, labels;
    load_mnist("data/mnist_train.csv", inputs, labels, 20);

    // Fully Connected Layer (입력 784개, 출력 10개)
    FCLayer fc(784, 10, 0.01);

    // 학습 루프
    for (int epoch = 0; epoch < 10; epoch++) {
        double total_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); i++) {
            total_loss += train(fc, inputs[i], labels[i]);
        }
        std::cout << "Epoch " << epoch << " Loss: " << total_loss / inputs.size() << std::endl;
    }

    return 0;
}*/