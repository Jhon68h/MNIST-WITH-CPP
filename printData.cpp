#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

auto printFuntion(vector<vector<float>> output_layer){
  cout << "\n filas de output: " << output_layer.size() << endl;
  cout << "\n columnas de output: " << output_layer[0].size() << endl;

  int rowsToPrint = std::min(10, static_cast<int>(output_layer.size()));
  int colsToPrint = std::min(10, static_cast<int>(output_layer[0].size()));

  for(int i = 0; i < rowsToPrint; i++){
    for(int j = 0; j < colsToPrint; j++){
      cout << fixed << setprecision(5) << setw(10) << output_layer[i][j] << "\t";
    }
    cout << endl;
  }
}