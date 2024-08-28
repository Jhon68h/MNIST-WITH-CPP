#include "include/backpropagation.hpp"
#include <cstddef>
#include <vector>

void BP::gradient(float learningRate, const vector<vector<float>>& weights){
    //Implementación del descenso del gradiente
    //formula
    //Wn+1 = W - γ∇f(W)
    //Wn+1  -> Peso nuevo
    //W     -> Peso Anterior
    //γ     -> Learning rate
    //∇f(W) -> Derivada parcial del error 
    
    

}

vector<float> derivateRelu(vector<float> input) {
    auto sizeInput = input.size(); 
    vector<float> derivate;
    
    for (size_t i = 0; i < sizeInput; i++) {
        derivate[i] = (input[i] > 0) ? 1.0f : 0.0f;
    }
    return derivate;
}