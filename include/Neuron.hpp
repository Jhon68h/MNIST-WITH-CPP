#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm> 

using namespace std;

class Neuron {
public:
    //Neuron() = default;
    Neuron(int next_layer_size, vector<vector<float>> inputNeurons);

//OBTENCIÓN DE PESOS Y BIAS
    vector<vector<float>> getWeight();
    vector<float> getBias();

//OPERACIONES DE MATRICES
    vector<vector<float>> multiply_perceptron(vector<vector<float>> A, vector<vector<float>> B);
    vector<vector<float>> transposeFunction(vector<vector<float>>& data);
    vector<vector<float>> nextLayer;

//OBTENCIÓN DE RESULTADOS
    int get_input_size_col() { return input_size_col;}
    int get_input_size_row() { return input_size_row;}
    int get_nextLayer_size_col() {return next_size_col;}
    int get_nextLayer_size_row() { return next_size_row;}
    vector<vector<float>> getWeightVector() const {return weight;};
    vector<vector<float>> getNextLayer() { return nextLayer; }
    vector<vector<float>> getMultiply_perceptron() { return operation; }

private:

    vector<vector<float>> inputNeurons;
    vector<vector<float>> data;
    vector<vector<float>> weights_for_operation;
    vector<vector<float>> weight;
    vector<vector<float>> A;
    vector<vector<float>> B;
    vector<vector<float>> operation;


    int next_layer_size;
    int input_size_col;
    int input_size_row;
    int next_size_col;
    int next_size_row;

};

