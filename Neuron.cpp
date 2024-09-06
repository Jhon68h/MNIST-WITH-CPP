#include "include/Neuron.hpp"


Neuron::Neuron(int next_layer_size, vector<vector<float>> inputNeurons)
 : inputNeurons(inputNeurons), next_layer_size(next_layer_size){

  if (inputNeurons.empty()) {    
    throw runtime_error("Input neurons are empty.");
  }

  input_size_col = inputNeurons[0].size();
  input_size_row = inputNeurons.size();


//No se le agrega bias ya que están iniciados en 0.

/*
genero un vector nextlayer del tamaño requerido para así realizar la operación en otra función
almacenando el resultado en el perceptron
*/
  nextLayer = vector<vector<float>> (input_size_col, vector<float>(next_layer_size, 0.0));

  next_size_col = nextLayer[0].size();
  next_size_row = nextLayer.size();

//********************DECLARACION DE PESOS**********************//

  weights_for_operation = getWeight();

  if(input_size_col != next_size_row){
    inputNeurons = transposeFunction(inputNeurons);
  }
  if(nextLayer.empty()){throw runtime_error("perceptron is empty");}

/*
  cout << "\n filas de la siguiente capa: " << nextLayer.size() << endl;
  cout << "\n columnas de la siguiente capa: " << nextLayer[0].size() << endl;
  cout << "\n input neurons size: " << inputNeurons.size() << endl;
  cout << "\n input neurons col: " << inputNeurons[0].size() << endl;
  cout << "\n pesos columnas: " << weights_for_operation[0].size() << endl;
  cout << "\n pesos filas: " << weights_for_operation.size() << endl;
*/

  multiply_perceptron(inputNeurons, weights_for_operation);

}

//**********FUNCION DE TRASNPOSICIÓN DE MATRICES**********//

vector<vector<float>> Neuron::transposeFunction(vector<vector<float>>& data){

  if (data.empty()){
    cout << "data empty";
    return {}; //devuelve un vector vacio
  }

  int rows = data.size();
  int cols = data[0].size();

  vector<vector<float>> transposedData(cols, vector<float>(rows));

  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      transposedData[j][i] = data[i][j]; 
    }
  }
  return transposedData;
}

//FUNCION PARA LA OBTENCION DE PESOS//

vector<vector<float>> Neuron::getWeight(){
  
  int input_size_for_weight = get_input_size_col();
  int output_size_for_weight = get_nextLayer_size_col();

  weight = vector<vector<float>>(input_size_for_weight, vector<float>(output_size_for_weight));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, sqrt(1.0 / input_size_for_weight));

  for(int i = 0; i < input_size_for_weight; i++){
    for(int j =0; j < output_size_for_weight; j++){
      weight[i][j] = d(gen);
    }
  }
  return weight;
}

vector<float> Neuron::getBias(){

    int size = get_nextLayer_size_col();

    vector<float> bias(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.001, 0.01);

    for (int i = 0; i < size; ++i) {
        bias[i] = d(gen);
    }

    return bias;
}

vector<vector<float>> Neuron::multiply_perceptron(vector<vector<float>> A, vector<vector<float>> B){

  int rowsA = A.size();
  int colsA = A[0].size();
  int rowsB = B.size();
  int colsB = B[0].size();

  if (colsA != rowsB) {
    throw runtime_error("Matrices cannot be multiplied due to incompatible dimensions.");
  }

/*Se crea un vector operación que realiza la multiplicacion entre dos vectores*/

  operation = vector<vector<float>> (rowsA, vector<float>(colsB, 0.0));
  vector<float> bias_operation = getBias();


  for (int i = 0; i < rowsA; ++i) {
    
    for (int j = 0; j < colsB; ++j) {
      for (int k = 0; k < colsA; ++k) {
        operation[i][j] += A[i][k] * B[k][j];
      }
      operation[i][j] += bias_operation[j];
    }
  }

  return operation;
}

