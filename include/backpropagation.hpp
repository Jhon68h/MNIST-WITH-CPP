#pragma once

#include <iostream>
#include <vector>

using namespace std;

class BP{

    public:
        BP();
        //Descenso del gradiente
        void gradient(float learningRate, const vector<vector<float>>& weights);
        vector<float> derivateRelu(vector<float> input);
        vector<vector<float>> derivateSoftmax(vector<vector<float>> input);
    private:

        float learningRate;
        const vector<vector<float>>& weights;

};     

