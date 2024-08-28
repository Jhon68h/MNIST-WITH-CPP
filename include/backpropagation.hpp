#pragma once

#include "../activation_functions.cpp"
#include <vector>

class BP{

    public:
        BP();
        //Descenso del gradiente
        void gradient(float learningRate, const vector<vector<float>>& weights);

    private:

        float learningRate;
        const vector<vector<float>>& weights;

};     

