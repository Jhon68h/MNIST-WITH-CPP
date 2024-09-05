#include <iostream>
#include <vector>
#include <cmath>
#include "loss_function.cpp"
#include "activation_functions.cpp"

using namespace std;

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


auto backpropagation(vector<float> input, vector<vector<float>> softmaxVector, vector<vector<int>> distributionVector){

//backpropagation ultima capa = derivateCostVsSoftmax * derivateSoftmaxLogits

    

}