#include "include/loss_functions.hpp"
#include <iterator>


//This function visualize the bigest probability and takes that position like
//the prediction
vector<int> Loss::prediction(vector<vector<float>> x){

    vector<int> predicted_numbers_list;

    for(auto& row : x){
        auto maxElement = max_element(row.begin(), row.end());//search the max element int the row
        auto predicted_numbers = distance(row.begin(), maxElement);//mark that position
        
        predicted_numbers_list.push_back(predicted_numbers);
    }
    return predicted_numbers_list;
}

vector<vector<int>> distributionVector(vector<int> labels){
    int cols = 10;
    int value = 1;
    auto rows = labels.size();
    vector<vector<int>> positionVector(rows, vector<int>(cols, 0));
    
    for (int i = 0; i < rows; i++) {
        int label = labels[i];
        if (label >= 0 && label < cols) { // Verifica que el label esté dentro de los límites
            positionVector[i][label] = 1; // Coloca un 1 en la posición 'label' de la fila 'i'
        }
    }

    return positionVector;
}

//{5}{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}


// cross entropy
// H(p, q) = -SUM i->M p(i)log(q(i))
// M = num of classes
// p(i) probabilidad verdadera del evento i
// q(i) probabilidad predicha por el modelo

