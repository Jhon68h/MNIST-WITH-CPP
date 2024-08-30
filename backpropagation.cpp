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

vector<float> BP::derivateRelu(vector<float> input) {
    auto sizeInput = input.size(); 
    vector<float> derivate;
    
    for (size_t i = 0; i < sizeInput; i++) {
        derivate[i] = (input[i] > 0) ? 1.0f : 0.0f;
    }
    return derivate;
}

vector<vector<float>> BP::derivateSoftmax(vector<vector<float>> softmaxVector, vector<vector<float>> distributionVector){
    /*se necesita calcular la derivada de la función de perdidad
    con respecto a los logits z_i que entran en la función de softmax
    */
    //Kronecker delta
    
    //Por el delta de Kronecjer [i = j] = 1 --->  soft(z_i)*(1-soft(z_i)) 

    //Por el delta [i != j] = 0 ---> -soft(z_i)*soft(z_j)

    //Entonces la derivada vendria siendo softmax(i) - y_i donde y_i es 1 para la clase
    //correcta y 0 para la incorrecta

/// QUE RECIBE?/////////////////////////////

    //recibe el valor de softmax
    //recibe el vector de valores correctos
    int softmaxSize = softmaxVector.size();
    int distributionSize = distributionVector.size();

    for(int i = 0; i < softmaxSize; i++){
        for (int j = 0; j < distributionSize; j++) {
        
        }
    }

}