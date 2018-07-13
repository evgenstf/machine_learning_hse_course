// Author: Evgenii Kazakov. Github: @evgenstf
#pragma once
#include <fstream>
#include <string>

namespace ml_workflow {

class Filter {
public:
  Filter(const std::string& raw_data_file) {
    download_data(raw_data_file);
  }

  void download_data(const std::string& raw_data_file) {
    std::ifstream file(raw_data_file);
    fields_ = read_line(file);
    data_.push_back(read_line(file));
    while (data_.back().size()) {
      data_.push_back(read_line(file));
    }
  }

  void filter() {
    for (auto data : data_) {
      filtered_data_.emplace_back();
      for (auto field : data) {
        bool digit = 1;
        for (auto c : field) {
          if (!('0' <= c && c <= '9')) {
            digit = 0;
          }
        }
        if (digit) {
          filtered_data_.back().push_back(std::stoi(field));
        } else {
          filtered_data_.back().push_back(0);
        }
      }
    }
  }

  void write_to_file(const std::string& file_name) {
    std::ofstream file(file_name);
    for (const auto& data : data_) {
      for (const auto& field : data) {
        file << field << ' ';
      }
      file << std::endl;
    }
  }

private:
  std::vector<std::string> fields_;
  std::vector<std::vector<std::string>> data_;
  std::vector<std::vector<double>> filtered_data_;
};

}  // namespace ml_workflow
