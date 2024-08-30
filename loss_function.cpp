#include "include/loss_functions.hpp"
#include <cmath>
#include <iterator>
#include <vector>
#define epsilon 1e-10f

Loss::Loss(vector<int> labels, const vector<vector<float>>& probabilities, vector<int> prediction){

    vector<vector<int>> distributionVectorVariable = Loss::distributionVector(labels);

    auto crossEntropy = Loss::cross_entropy(distributionVectorVariable, probabilities);
}

/*
//This function visualize the largest probability and takes that position as the prediction
vector<int> Loss::prediction(vector<vector<float>> x){

    vector<int> predicted_numbers_list;

    for(auto& row : x){
        auto maxElement = max_element(row.begin(), row.end());//search the max element int the row
        auto predicted_numbers = distance(row.begin(), maxElement);//mark that position
        
        predicted_numbers_list.push_back(predicted_numbers);
    }
    return predicted_numbers_list;
}
*/

vector<vector<int>> Loss::distributionVector(vector<int> labels){
    //La función tiene que leer el label y en un vector de 0 de 10 posiciones
    //pondrá un 1 en la posicion que el label marque

    //EJEMPLO -> {5}{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}

    int cols = 10;
    int value = 1;
    auto rows = labels.size();
    vector<vector<int>> positionVector(rows, vector<int>(cols, 0));
    
    for (int i = 0; i < rows; i++) {
        int label = labels[i];
        if (label >= 0 && label < cols) { // Verifica que el label esté dentro de los límites
            positionVector[i][label] = 1; // Coloca un 1 en la posición 'label' de la fila 'i'
        }
    }

    return positionVector;
}


float Loss::cross_entropy(const vector<vector<int>>& distributionVector, const vector<vector<float>>& predictions){
    //Se ingresa el distributionVector, este vector fue implementado en la anterior
    //función, que como se explicó, es un vector que determina la posición
    //real del label
    auto distributionVectorSize = distributionVector.size();

    float sum = 0.0f;

    #pragma omp parallel for reduction(-:sum) // Paralelización del bucle
    for(int i = 0; i < distributionVectorSize; i++) {
        for(int j = 0; j < 10; j++) {
            // cross entropy
            // H(p, q) = -SUM i->M p(i)log(q(i))
            // M = num of classes
            // p(i) probabilidad verdadera del evento i
            // q(i) probabilidad predicha por el modelo
            sum -= distributionVector[i][j] * log(predictions[i][j] + epsilon);
        }
    }

    return sum;
}