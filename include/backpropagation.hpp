#pragma once

#include <iostream>
#include <ratio>
#include <vector>
#include "../activation_functions.cpp"
#include "loss_functions.hpp"

using namespace std;

class BP{

    public:
        BP();
        //Descenso del gradiente
        void gradient(float learningRate, vector<vector<float>> softmax, vector<vector<int>> labelsVector,const vector<vector<float>>& weights);
        vector<float> derivateRelu(vector<float> input);
        vector<vector<float>> derivateSoftmax(vector<vector<float>> input, vector<vector<int>> distributionVector);


    private:

        float learningRate;
        const vector<int>& labels;
        const vector<vector<float>>& weights;
        vector<vector<float>> softmax;
        vector<vector<int>> labelsVector;
};     

