#include "include/Backpropagation.hpp"

//La lógica principal es saber "COMO VARIA EL COSTE RESPECTO A LOS PARAMETROS DE LA RED"
//  ∂C/∂w * ∂C/∂b
//  pesos * bias

//entonces si quiero saber como la variación de los pesos afecta a la función de costo
//tengo que aplicar la regla de la cadena para sacar las derivadas de esta
//función compuesta 

/*
                               C(A(Z))  

                               C -> función de costo
                               A -> función de activación
                               Z -> sumatoría de la ecuacion de perceptrón
*/

//Si tenemos una cantidad de capas L, inicia desde la ultíma, es decir L

//Entonces siendo x = w v b
//∂C/∂x = ∂C/∂A * ∂A/∂Z * ∂Z/∂x
//Todo esto para la última capa 
//Siendo ∂C/∂w = (a_{i})^L-1 y ∂C/∂b = 1


vector<vector<float>> Backpropagation::derivateSoftmaxLogits(const vector<vector<float>>& softmaxVector) {
    //La derivada de la función de activación respecto a los logits 
    
    //Kronecker delta
    
    //Por el delta de Kronecjer [i = j] = 1 --->  soft(z_i)*(1-soft(z_i)) 

    //Por el delta [i != j] = 0 ---> -soft(z_i)*soft(z_j)
    size_t row = softmaxVector.size();
    size_t col = softmaxVector[0].size();
    vector<vector<float>> gradient(row, vector<float>(col, 0.0f));

    // Calculamos la derivada para la diagonal y no diagonal en un solo bucle
    for (int k = 0; k < row; k++) {
        for (size_t i = 0; i < col; ++i) {
            // Diagonal i == j
            gradient[k][i] = softmaxVector[k][i] * (1.0f - softmaxVector[k][i]);

            // No diagonal: acumulamos los términos de i != j
            for (size_t j = 0; j < col; ++j) {
                if (i != j) {
                    gradient[k][i] -= softmaxVector[k][i] * softmaxVector[k][j];
                }
            }
        }
    }
    
    return gradient; //DIMENSIONES -> IGUALES A SOFTMAX VECTOR (60000, 10)
}


vector<vector<float>> Backpropagation::derivateCostVsSoftmax(vector<vector<float>> softmaxVector, 
                                            vector<vector<float>> distributionVector){
    
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
    size_t row = softmaxVector.size();
    size_t col = softmaxVector[0].size();

    vector<vector<float>> derivateCost(row, vector<float>(col, 0.0f));
    
    for(size_t i = 0; i < row; i++){
        for (size_t j = 0; j < col; j++) {
            derivateCost[i][j] = softmaxVector[i][j] - distributionVector[i][j];
        }
    }

    return derivateCost;//DIMENSIONES IGUAL 60000, 10

}


vector<vector<float>> Backpropagation::deltaValue(const vector<vector<float>>& softmaxVector, 
                vector<vector<float>> distributionVector){

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

    return delta;//DIMENSIONES IGUAL 60000, 10

}

vector<vector<float>> Backpropagation::outputBackPropagation(vector<vector<float>> output, 
                            const vector<vector<float>> &softmaxVector, 
                            vector<vector<float>> distributionVector, bool Bias = false){
                                
    //Se realiza el backpropagation, sin embargo esto solo funciona
    //para la última capa, con los ultimos pesos

    //La formula para los Bias es igual, sin embargo la derivada parcial 
    //respecto a Bias es 1, entonces en el caso Bias se devuelve el mismo
    //delta
    vector<vector<float>> delta = deltaValue(softmaxVector, distributionVector);

    if (!Bias) {

        int deltaRow = delta.size();
        int deltaCol = delta[0].size();
        int outputCol = output[0].size();

        vector<vector<float>> gradient(deltaRow, vector<float>(outputCol, 0.0f));

        for (size_t i = 0; i < deltaRow; i++) {
            for (size_t j = 0; j < outputCol; j++) {
                for (size_t k = 0; k < deltaCol; k++) {
                    gradient[i][j] += delta[i][k] * output[k][j];
                }   
            }
        }
    
        return gradient;
    }
    
    return delta;
}


vector<vector<float>> Backpropagation::hiddenBackPropagation(vector<vector<float>> output, 
                                  vector<vector<float>> softmaxVector, 
                                  vector<vector<float>> distributionVector, 
                                  bool Bias = false){
    // Para las capas ocultas se usa la siguiente formula
    //para Delta-1 --> W * Delta * ∂a-1/∂z-1
    //y completa entonces --> Delta-1 * a-2

    auto delta = deltaValue(softmaxVector, distributionVector);
    
    

}