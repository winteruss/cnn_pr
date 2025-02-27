#ifndef LOADCSV_H
#define LOADCSV_H

#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "matrix.h"

class DataLoader {
  public:
    static void loadCSV(const std::string& filename, std::vector<Matrix>& images, std::vector<Matrix>& labels) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open: " << filename << std::endl;
            return;
        }
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<double> image_data;
            std::vector<double> label_data(10, 0.0); // One-Hot Vector (0~9)

            std::string value;
            int idx = 0;
            int label_value;

            while (std::getline(ss, value, ',')) {
                if (idx == 0) {
                    label_value = std::stoi(value);
                    label_data[label_value] = 1.0;
                } else {
                    image_data.push_back(std::stod(value) / 255.0);
                }
                idx++;
            }

            images.emplace_back(28, 28, image_data);
            labels.emplace_back(10, 1, label_data);
        }
    }
};

#endif