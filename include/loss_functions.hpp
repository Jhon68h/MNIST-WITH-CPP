#pragma once

#include <iostream>
#include <iterator>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class Loss {

    public:

        Loss(int epoch, const vector<int>& labels, const vector<vector<float>>& probabilities);

        // Función que convierte los labels en un vector de distribución binario
        vector<vector<int>> distributionVector(const vector<int>& labels);
        // Función que calcula la entropía cruzada dada una distribución y predicciones
        double cross_entropy(const vector<vector<int>>& distributionVector, vector<vector<float>> prediction);

    private:
        const vector<int>& labels;  // Referencia constante a los labels
        const vector<vector<float>>& probabilities;  // Referencia constante a las probabilidades
        int epoch; //Epocas 
        vector<float> lossValues; //Vector para almacenar el valor de la funcion de perdidad en cada epoca

};
    