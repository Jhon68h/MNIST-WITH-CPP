#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

bool checkProbabilities(const std::vector<std::vector<float>>& x) {
    for (const auto& row : x) {
        float sum = std::accumulate(row.begin(), row.end(), 0.0f);
        if (std::abs(sum - 1.0f) > 1e-5) {
            return false;
        }
    }
    return true;
}

