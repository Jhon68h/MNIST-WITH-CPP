#ifndef LOSS_HPP
#define LOSS_HPP

#include <iostream>
#include <iterator>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class Loss{

    public:

        Loss(vector<int> labels, const vector<vector<float>>& probabilities, vector<int> predictions);
        
        //vector<int> prediction(vector<vector<float>> x);
        vector<vector<int>> distributionVector(vector<int> labels);
        float cross_entropy(const vector<vector<int>>& distributionVector, const vector<vector<float>>& predictions);

    private:

        vector<int> labels;
        vector<vector<float>> probabilities;
        vector<vector<float>> x;
        vector<vector<float>> predictions;
        
};

#endif