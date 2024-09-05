#ifndef READ_PLOT_HPP
#define READ_PLOT_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp> 

using namespace std;

class ReadData {
public:
    //ReadData() = default;

    ReadData(const string &filename);

    //vector<vector<float>> convertToImage(const vector<float> &row);
    //void plotImage(const vector<vector<float>> &image);

    vector<vector<float>> readCSV(const string &filename);

    // Método para obtener las etiquetas
    const vector<int>& getLabels() const {
        return labels;
    }

private:

    // Datos leídos del archivo CSV
    vector<vector<float>> data;
    vector<int> labels;  // Vector para almacenar las etiquetas
};

ReadData::ReadData(const string &filename) {
    //data = readCSV(filename);
    /*for (const auto &row : data) {
        vector<vector<float>> image = convertToImage(row);
        plotImage(image);
    }*/
}

vector<vector<float>> ReadData::readCSV(const string &filename) {
    vector<vector<float>> data;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo " << filename << endl;
        return data;
    }

    string line;

    if (getline(file, line)) {
        cout << " " <<endl;
    } else {
        cerr << "Error: No se pudo leer el encabezado del archivo CSV." << endl;
        return data;
    }

    while (getline(file, line)) {
        vector<float> row;
        stringstream ss(line);
        string value;

        // Leer la etiqueta (primera columna) y almacenarla en `labels`
        if (getline(ss, value, ',')) {
            try {
                value.erase(0, value.find_first_not_of(' ')); 
                value.erase(value.find_last_not_of(' ') + 1);

                if (!value.empty()) {
                    labels.push_back(stoi(value));
                }
            } catch (const invalid_argument &e) {
                cerr << "Valor no válido encontrado en la etiqueta: " << value << endl;
            }
        }

        // Leer los valores de las características (columnas restantes)
        while (getline(ss, value, ',')) {
            try {
                value.erase(0, value.find_first_not_of(' ')); 
                value.erase(value.find_last_not_of(' ') + 1);

                // Solo intenta convertir si no está vacío
                if (!value.empty()) {
                    row.push_back(stoi(value));
                }
            } catch (const invalid_argument &e) {
                cerr << "Valor no válido encontrado: " << value << endl;
                // Manejo de errores si es necesario
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    cout << "\nNúmero de filas leídas del csv: " << data.size() << endl;
    cout << "tamaño fila del csv: " << data[0].size() << endl;

    return data;
}


/*METODOS PARA PLOTEAR LA IMAGEN

ACTIVAR LA FUNCION EN LA CLASE
*/

/*vector<vector<float>> ReadData::convertToImage(const vector<float> &row) {
    vector<vector<float>> image(28, vector<float>(28));
    for (size_t i = 1; i < row.size(); ++i) {
        int pixelValue = row[i];
        image[(i - 1) / 28][(i - 1) % 28] = pixelValue;
    }
    return image;
}

void ReadData::plotImage(const vector<vector<float>> &image) {
    cv::Mat img(28, 28, CV_8U);
    
    for (size_t i = 0; i < 28; ++i) {
        for (size_t j = 0; j < 28; ++j) {
            img.at<uchar>(i, j) = image[i][j];
        }
    }
    
    cv::imshow("MNIST Image", img);
    cv::waitKey(0);
}
*/

#endif 
