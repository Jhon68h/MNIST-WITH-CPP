#include <cmath>
#include <algorithm> 
#include <vector>


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
    for (auto& row : input) {
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