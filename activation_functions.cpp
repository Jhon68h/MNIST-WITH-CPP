#include <cmath>
#include <algorithm> 
#include <cstddef>
#include <vector>
#include <iostream>

using namespace std;

//funcion relu

void relu(vector<vector<float>>& input) {
    for (auto& row : input) {
        for (auto& value : row) {
            value = max(0.0f, value);
        }
    }
}

//aplicacion de la funcion softmax

void softmax(vector<vector<float>>& input) {
    //e^zj/(Sum{k=1}->{K} e^z_{k}) con j -> 1,..,K
    for (auto& row   : input) {
        float max_val = *max_element(row.begin(), row.end());
        float sum_exp = 0.0f;

        for (auto& value : row) {
            value = std::exp(value - max_val);
            sum_exp += value;
        }

        for (auto& value : row) {
            value /= sum_exp;
        }
    }
}

vector<vector<float>> derivateRelu(vector<vector<float>> input) {
    auto sizeInput = input.size(); 
    vector<vector<float>> derivate;
    
    for (size_t i = 0; i < sizeInput; i++) {
        for(size_t j = 0; j < input[0].size(); j++){
            derivate[i][j] = (input[i][j] > 0) ? 1.0f : 0.0f;
        }
    }
    return derivate;
}

