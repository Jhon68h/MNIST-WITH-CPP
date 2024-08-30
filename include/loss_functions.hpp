#pragma once

#include <iostream>
#include <iterator>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class Loss {

    public:
        // Constructor que inicializa los miembros de referencia 'labels' y 'probabilities'
        Loss(const vector<int>& labels, const vector<vector<float>>& probabilities);

        // Función que convierte los labels en un vector de distribución binario
        vector<vector<int>> distributionVector(const vector<int>& labels);
        // Función que calcula la entropía cruzada dada una distribución y predicciones
        float cross_entropy(const vector<vector<int>>& distributionVector, vector<vector<float>> prediction);

    private:
        const vector<int>& labels;  // Referencia constante a los labels
        const vector<vector<float>>& probabilities;  // Referencia constante a las probabilidades
        // Las siguientes variables fueron eliminadas ya que no eran necesarias:
        // vector<vector<float>> x;
        // vector<vector<float>> prediction;
};
