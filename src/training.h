#ifndef TRAINING_H
#define TRAINING_H

#include "matrix.h"
#include "fc.h"
#include "activation.h"
#include "loss.h"

/*inline double train(FCLayer& fc, const Matrix& input, const Matrix& target) {

    Matrix logits = fc.forward(input);
    Matrix probabilities = Softmax(logits);

    double loss = CrossEntropyLoss(probabilities, target);

    Matrix d_output = probabilities - target;
    fc.backward(input, d_output);

    return loss;
}*/

#endif
