#include "include/read_plot.hpp"
#include "include/Neuron.hpp"
#include "activation_functions.cpp"
#include "sumProbabilities.cpp"
#include "loss_function.cpp"
#include <iomanip>
#include "shuffle.cpp"
#include "printData.cpp"


int main(){
  
  //LECTURA DE DATOS
  string testData = "/home/jhonatan/Documents/C++/MNIST_TRY/dataset/mnist_test_normalized.csv";
  string trainData = "/home/jhonatan/Documents/C++/MNIST_TRY/dataset/mnist_train_normalized.csv";
  
  ReadData rd(trainData);

  vector<vector<float>> data_from_readcsv = rd.readCSV(trainData);
  
  vector<int> labels = rd.getLabels();
////////////////////////////////////////////////////////////////////////
  //CAPA OCULTA
////////////////////////////////////////////////////////////////////////
  int hidden_layer_size = 16;

  Neuron neuron(hidden_layer_size, data_from_readcsv);

  vector<vector<float>> operation = neuron.getMultiply_perceptron();
  
  relu(operation);

  cout << "\n filas de capa oculta: " << operation.size() << endl;
  cout << "\n columnas de capa oculta: " << operation[0].size() << endl;

  //////////////////////////////////////////////////////////////////
  //CAPA DE SALIDA
  //////////////////////////////////////////////////////////////////
  int output_layer_size = 10;

  Neuron neuron2(output_layer_size, operation);

  vector<vector<float>> output_layer = neuron2.getMultiply_perceptron();
  softmax(output_layer);
  
  //Imprimir las 10 primeras filas de resultados
  printFuntion(output_layer);

  vector<int> predictionOutput = prediction(output_layer);

  printFuntion(predictionOutput);

  if (checkProbabilities(output_layer)) {
      cout << "\nLas sumas de probabilidades da 1." << endl;
  } else {
      cout << "\nError: Algunas filas no suman 1." << endl;
  }

  return 0;
}
