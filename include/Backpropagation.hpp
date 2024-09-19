#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
#include <cmath>    
#include <algorithm>
#include "../loss_function.cpp"
#include "../activation_functions.cpp"

class Backpropagation {
public:
    // Constructor por defecto
    Backpropagation() = default;

    // Método para calcular la derivada de los logits de softmax.
    static vector<vector<float>> derivateSoftmaxLogits(const vector<vector<float>>& softmaxVector);

    // Método para calcular la derivada del coste respecto al valor de softmax.
    static vector<vector<float>> derivateCostVsSoftmax(const vector<vector<float>>& softmaxVector, 
                                                                 const vector<vector<float>>& distributionVector);

    // Método para calcular el valor de delta (cambio) de la última capa.
    static vector<vector<float>> deltaValue(const vector<vector<float>>& softmaxVector, 
                                                      const vector<vector<float>>& distributionVector);

    // Método para realizar backpropagation en la última capa, actualizando pesos o bias.
    static vector<vector<float>> outputBackPropagation(const vector<vector<float>>& output, 
                                                                 const vector<vector<float>>& softmaxVector, 
                                                                 const vector<vector<float>>& distributionVector, 
                                                                 bool Bias = false);

    // Método para realizar backpropagation en las capas ocultas.
    static vector<vector<float>> hiddenBackPropagation(const vector<vector<float>>& output, 
                                                                 const vector<vector<float>>& softmaxVector, 
                                                                 const vector<vector<float>>& distributionVector, 
                                                                 bool Bias = false);
};
