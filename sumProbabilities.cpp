#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

using namespace std;

bool checkProbabilities(const vector<vector<float>>& x) {
    for (const auto& row : x) {
        float sum = accumulate(row.begin(), row.end(), 0.0f);
        if (abs(sum - 1.0f) > 1e-5) {
            return false;
        }
    }
    return true;
}

vector<int> prediction(vector<vector<float>> x){

    vector<int> predicted_numbers_list;

    for(auto& row : x){
        auto maxElement = max_element(row.begin(), row.end());//search the max element int the row
        auto predicted_numbers = distance(row.begin(), maxElement);//mark that position
        
        predicted_numbers_list.push_back(predicted_numbers);
    }
    return predicted_numbers_list;
}