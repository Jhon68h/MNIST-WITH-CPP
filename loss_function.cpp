#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

#define epsilon 1e-10f

using namespace std;

/*
Loss::Loss(int epoch, const vector<int>& labels, const vector<vector<float>>& probabilities)
    : epoch(epoch), labels(labels), probabilities(probabilities) {  // Usa la lista de inicialización

    // Utiliza los miembros de la clase en lugar de crear nuevas variables
    vector<vector<int>> distributionVectorVariable = distributionVector(labels);

    // Calcula la entropía cruzada usando los vectores distribuidos y las probabilidades
    lossValues.resize(epoch);
    for(int i = 0; i < epoch; i++){
        auto crossEntropy = cross_entropy(distributionVectorVariable, probabilities);
        lossValues[i] = crossEntropy;
    }

}*/



vector<vector<int>> distributionVector(const vector<int>& labels){
    // La función tiene que leer el label y en un vector de 0 de 10 posiciones
    // pondrá un 1 en la posición que el label marque

    // EJEMPLO -> {5} -> {0, 0, 0, 0, 0, 1, 0, 0, 0, 0}

    int cols = 10;
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

double cross_entropy(const vector<vector<int>>& distributionVector, vector<vector<float>> prediction) {
    // Se ingresa el distributionVector, este vector fue implementado en la anterior
    // función, que como se explicó, es un vector que determina la posición
    // real del label

    //EL vector de prediction es el vector que arroja las probabilidades obtenidas
    //no el que da el numero que predijo sino las probabilidades
    
    auto distributionVectorSize = distributionVector.size();

    double sum = 0.0f;

    #pragma omp parallel for reduction(-:sum) // Paralelización del bucle
    for(size_t i = 0; i < distributionVectorSize; i++) {
        for(size_t j = 0; j < 10; j++) {
            // Entropía cruzada
            // H(p, q) = -SUM i->M p(i)log(q(i))
            // M = número de clases
            // p(i) = probabilidad verdadera del evento i
            // q(i) = probabilidad predicha por el modelo
            sum -= distributionVector[i][j] * log(prediction[i][j] + epsilon);
        }
    }

    return sum / distributionVectorSize;
}

auto derivateSoftmaxLogits(const vector<vector<int>>& softmaxVector, const vector<vector<float>>& prediction) {
    //La derivada de la función de activación respecto a los logits 
    
    size_t row = prediction.size();
    size_t col = prediction[0].size();
    vector<vector<float>> gradient(row, vector<float>(col));

    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < col; ++j) {
            if(i == j){
                gradient[i][j] = softmaxVector[i][j] * (1 - softmaxVector[i][j]);
            }else{
                gradient[i][j] -= softmaxVector[i][j] * softmaxVector[i][j];
            }
        }
    }

    return gradient;
}
