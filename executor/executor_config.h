// Author: Evgenii Kazakov. Github: @evgenstf
#pragma once
#include <string>
#include <vector>
#include <fstream>

namespace ml_workflow {

struct ExecutorConfig {
  ExecutorConfig(const std::string& file_name) {
    std::ifstream file(file_name);
    file >> nearest_cnt >> need_to_survive >> rows >> columns;
    data.resize(rows, std::vector<double>(columns, 0));
    for (size_t row = 0; row < rows; ++row) {
      for (size_t column = 0; column < columns; ++column) {
        file >> data[row][column];
      }
    }

    indicators.resize(columns);
    for (auto& indicator : indicators) {
      file >> indicator;
    }
  }

  void write_to_file(const std::string& file_name) {
    std::ofstream file(file_name);
    file << nearest_cnt << ' ' << need_to_survive << ' ' << rows << ' ' << columns;
    file << '\n';
    for (size_t row = 0; row < rows; ++row) {
      for (size_t column = 0; column < columns; ++column) {
        file << data[row][column] << " \n"[column == columns - 1];
      }
    }
  }

  size_t nearest_cnt;
  size_t need_to_survive;
  size_t rows;
  size_t columns;
  std::vector<std::vector<double>> data;
  std::vector<int> indicators;
};

}  // namespace ml_workflow
