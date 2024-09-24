#include "include/read_plot.hpp"
#include "include/Neuron.hpp"
#include "sumProbabilities.cpp"
#include <iomanip>
#include <vector>
#include "shuffle.cpp"
#include "printData.cpp"
#include "Backpropagation.cpp"

#define learning_rate 0.01f

int main() {
    // Archivos de datos
    const std::string testDataPath = "/home/jhonatan/Documents/C++/MNIST_TRY/dataset/mnist_test_normalized.csv";
    const std::string trainDataPath = "/home/jhonatan/Documents/C++/MNIST_TRY/dataset/mnist_train_normalized.csv";
  
    // Lectura de datos
    ReadData rd(trainDataPath);
    std::vector<std::vector<float>> data = rd.readCSV(trainDataPath);
    std::vector<int> labels = rd.getLabels();

    // Configuraciones de capas
    const int hiddenLayerSize = 16;
    const int outputLayerSize = 10;

    // Capa oculta
    Neuron hiddenLayer(hiddenLayerSize, data);
    std::vector<std::vector<float>> hiddenLayerOutput = hiddenLayer.getMultiply_perceptron();
    relu(hiddenLayerOutput);  // Aplicar función de activación ReLU

    std::cout << "\nFilas de capa oculta: " << hiddenLayerOutput.size() << std::endl;
    std::cout << "Columnas de capa oculta: " << hiddenLayerOutput[0].size() << std::endl;

    // Capa de salida
    Neuron outputLayer(outputLayerSize, hiddenLayerOutput);
    std::vector<std::vector<float>> output = outputLayer.getMultiply_perceptron();
    softmax(output);  // Aplicar función softmax

    // Imprimir las 10 primeras filas de resultados
    printFuntion(output);
    
    // Predicciones
    std::vector<int> predictions = prediction(output);

    // One-hot encoding
    std::vector<std::vector<float>> oneHotEncoding = distributionVector(labels);

    // Cálculo de la entropía cruzada
    double crossEntropy = cross_entropy(oneHotEncoding, output);
    std::cout << "\nCross entropy: " << crossEntropy << std::endl;

    //backpropagation
    // vector<vector<float>> outputWeight = outputLayer.getWeightVector();
    // auto outputBack = outputBackPropagation(outputWeight, output, oneHotEncoding);
    // printFuntion(outputBack);
    
    // Imprimir predicciones
    printFuntion(predictions);

    

    // Verificar que las sumas de probabilidades sean 1
    if (checkProbabilities(output)) {
        std::cout << "\nLas sumas de probabilidades son correctas (suman 1)." << std::endl;
    } else {
        std::cout << "\nError: Algunas filas no suman 1." << std::endl;
    }

    return 0;
}
