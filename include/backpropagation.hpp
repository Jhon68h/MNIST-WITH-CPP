#pragma once

#include <iostream>
#include <vector>
#include "../activation_functions.cpp"
#include "loss_functions.hpp"

using namespace std;

class BP{

    public:
        BP();
        //Descenso del gradiente
        void gradient(float learningRate, const vector<vector<float>>& weights);
        vector<float> derivateRelu(vector<float> input);
        vector<vector<float>> derivateSoftmax(vector<vector<float>> input);

        vector<vector<int>> getDistribution(const vector<int>& labels){
            return Loss::distributionVector(labels);
        }

    private:

        float learningRate;
        const vector<int>& labels;
        const vector<vector<float>>& weights;

};     

