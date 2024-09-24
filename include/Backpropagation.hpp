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
    // Constructor
    Backpropagation(const vector<vector<float>> output, const vector<vector<float>> softmaxVector, const vector<vector<float>> distributionVector);

    // Método para calcular la derivada de los logits de softmax.
    vector<vector<float>> derivateSoftmaxLogits(const vector<vector<float>> softmaxVector);

    // Método para calcular la derivada del coste respecto al valor de softmax.
    vector<vector<float>> derivateCostVsSoftmax(const vector<vector<float>> softmaxVector, 
                                                                 const vector<vector<float>> distributionVector);

    // Método para calcular el valor de delta (cambio) de la última capa.
    vector<vector<float>> deltaValue(const vector<vector<float>> softmaxVector, 
                                                      const vector<vector<float>> distributionVector);

    // Método para realizar backpropagation en la última capa, actualizando pesos o bias.
    vector<vector<float>> outputBackPropagation(const vector<vector<float>> output, const vector<vector<float>> delta, bool Bias);   
                                                    // Método para realizar backpropagation en las capas ocultas.
    vector<vector<float>> hiddenBackPropagation(const vector<vector<float>> output, 
                                                                 const vector<vector<float>> softmaxVector, 
                                                                 const vector<vector<float>> distributionVector, 
                                                                 bool Bias);

private:
    const vector<vector<float>> output;
    const vector<vector<float>> softmaxVector;
    const vector<vector<float>> distributionVector;
    const vector<vector<float>> delta;
    bool Bias = false;
    
};
