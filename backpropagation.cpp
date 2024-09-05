#include <cstddef>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "loss_function.cpp"
#include "activation_functions.cpp"


using namespace std;

vector<vector<float>> derivateSoftmaxLogits(const vector<vector<float>>& softmaxVector);
vector<vector<float>> derivateCostVsSoftmax(vector<vector<float>> softmaxVector, vector<vector<int>> distributionVector);
auto backpropagation(const vector<vector<float>>& softmaxVector, const vector<vector<float>>& prediction,vector<vector<int>> distributionVector);


//La lógica principal es saber "COMO VARIA EL COSTE RESPECTO A LOS PARAMETROS DE LA RED"
//  ∂C/∂w * ∂C/∂b
//  pesos * bias

//entonces si quiero saber como la variación de los pesos afecta a la función de costo
//tengo que aplicar la regla de la cadena para sacar las derivadas de esta
//función compuesta 
//                             C(A(Z))
//C -> función de costo
//A -> función de activación
//Z -> sumatoría de la ecuacion de perceptrón

//Si tenemos una cantidad de capas L, inicia desde la ultíma, es decir L

//Entonces siendo x = w v b
//∂C/∂x = ∂C/∂A * ∂A/∂Z * ∂Z/∂x
//Todo esto para la última capa 
//Siendo ∂C/∂w = (a_{i})^L-1 y ∂C/∂b = 1


vector<vector<float>> derivateSoftmaxLogits(const vector<vector<float>>& softmaxVector) {
    //La derivada de la función de activación respecto a los logits 
    
    //Kronecker delta
    
    //Por el delta de Kronecjer [i = j] = 1 --->  soft(z_i)*(1-soft(z_i)) 

    //Por el delta [i != j] = 0 ---> -soft(z_i)*soft(z_j)
    size_t row = softmaxVector.size();
    size_t col = softmaxVector[0].size();
    vector<vector<float>> gradient(row, vector<float>(col, 0.0f));

    for(int k = 0; k < row; k++){
        for (size_t i = 0; i < col; ++i) {
            for (size_t j = 0; j < col; ++j) {
                if(i == j){
                    gradient[k][i] = softmaxVector[k][i] * (1.0f - softmaxVector[k][i]);
                }else{
                    gradient[k][i] -= softmaxVector[k][i] * softmaxVector[k][j];
                }
            }
        }
    }
    return gradient;
}

vector<vector<float>> derivateCostVsSoftmax(vector<vector<float>> softmaxVector, vector<vector<int>> distributionVector){
    
    /*se necesita calcular la derivada de la función de perdida
    con respecto a los logits z_i que entran en la función de softmax
    */

    
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

    for(size_t i = 0; i < row; i++){
        for(size_t j = 0; j < col; j++){
            derivateVector[i][j] = softmaxVector[i][j] - distributionVector[i][j];
        }
    }

    return derivateVector;

}

auto deltaValue(const vector<vector<float>>& softmaxVector, const vector<vector<float>>& prediction,vector<vector<int>> distributionVector, vector<vector<float>> weight){

//backpropagation ultima capa = derivateCostVsSoftmax * derivateSoftmaxLogits
    vector<vector<float>> x = derivateSoftmaxLogits(softmaxVector);
    vector<vector<float>> y = derivateCostVsSoftmax(softmaxVector, distributionVector);
    
    int row = x.size();
    int col = x[0].size();

    auto delta = vector<vector<float>> (row, vector<float> (col, 0.0f));
    
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            delta[i][j] = x[i][j] * y[i][j];   
        }
    }

}

