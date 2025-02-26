#ifndef LOAD_CSV_H
#define LOAD_CSV_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "matrix.h"

inline void load_mnist(const std::string& filename, std::vector<Matrix>& inputs, std::vector<Matrix>& labels, int sample_size = 1000) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    std::string line;
    int count = 0;
    
    while (std::getline(file, line) && count < sample_size) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> pixels;
        
        // 첫 번째 값 (정답 레이블)
        std::getline(ss, value, ',');
        int label = std::stoi(value);

        // 나머지 28x28 픽셀 값 (0~255 정규화)
        while (std::getline(ss, value, ',')) {
            pixels.push_back(std::stod(value) / 255.0);
        }

        // 입력 벡터 (28x28 → 784x1)
        Matrix input(784, 1);
        for (int i = 0; i < 784; i++) {
            input.data[i][0] = pixels[i];
        }

        // 정답 벡터 (One-Hot Encoding, 10 클래스)
        Matrix output(10, 1);
        output.data[label][0] = 1.0;

        inputs.push_back(input);
        labels.push_back(output);
        count++;
    }
    file.close();
}

#endif