#include <cmath>
#include <algorithm> 
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

vector<float> derivateRelu(vector<float> input) {
    auto sizeInput = input.size(); 
    vector<float> derivate;
    
    for (size_t i = 0; i < sizeInput; i++) {
        derivate[i] = (input[i] > 0) ? 1.0f : 0.0f;
    }
    return derivate;
}

vector<vector<float>> derivateSoftmax(vector<vector<float>> softmaxVector, vector<vector<int>> distributionVector){
    
    /*se necesita calcular la derivada de la función de perdidad
    con respecto a los logits z_i que entran en la función de softmax
    */
    //Kronecker delta
    
    //Por el delta de Kronecjer [i = j] = 1 --->  soft(z_i)*(1-soft(z_i)) 

    //Por el delta [i != j] = 0 ---> -soft(z_i)*soft(z_j)
    
    //Entonces la derivada vendria siendo softmax(i) - y_i donde y_i es 1 para la clase
    //correcta y 0 para la incorrecta

    /* Se necesita calcular la derivada de la función de pérdida
    con respecto a los logits z_i que entran en la función de softmax. */

    ///////////////////////////QUE RECIBE?/////////////////////////////

    //recibe el valor de softmax
    //recibe el vector de valores correctos
    auto row = softmaxVector.size();
    auto col = softmaxVector[0].size();

    vector<vector<float>> derivateVector(row, vector<float>(col, 0));

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            derivateVector[i][j] = softmaxVector[i][j] - distributionVector[i][j];
        }
    }

    return derivateVector;

}