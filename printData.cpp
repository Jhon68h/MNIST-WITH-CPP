#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

auto printFuntion(vector<vector<float>> x){
  cout << "\n filas: " << x.size() << endl;
  cout << "\n columnas: " << x[0].size() << endl;

  int rowsToPrint = std::min(10, static_cast<int>(x.size()));
  int colsToPrint = std::min(10, static_cast<int>(x[0].size()));

  for(int i = 0; i < rowsToPrint; i++){
    for(int j = 0; j < colsToPrint; j++){
      cout << fixed << setprecision(5) << setw(10) << x[i][j] << "\t";
    }
    cout << endl;
  }
}

auto printFuntion(vector<int> x){
  cout << "\n Predicciones" << endl;
  for(int i = 0; i < 10; i++){
    cout << x[i] << "\n";
  }
  cout << endl;
}