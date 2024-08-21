#include <random>
#include <algorithm>
#include <vector>
#include <cmath>

void shuffle(std::vector<std::vector<float>> x){
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(x.begin(), x.end(), g);
}

