// Author: Evgenii Kazakov. Github: @evgenstf
#pragma once
#include "executor_config.h"
#include <algorithm>
#include <iostream>

namespace ml_workflow {

namespace {

double dist(const std::vector<double> a, const std::vector<double> b) {
  double result = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    auto diff = a[i] - b[i];
    result += diff * diff;
  }
  return result;
}

auto find_nearest_known_items(const std::vector<double>& item,
    const ExecutorConfig& config) {
  std::vector<std::pair<double, int>> qwe;
  for (size_t i = 0; i < config.data.size(); ++i) {
    qwe.emplace_back(dist(config.data[i], item), i);
  }
  std::sort(qwe.begin(), qwe.end());
  std::clog << "nearest items: \n";
  for (const auto asd : qwe) {
    std::clog << asd.first << ' ' << asd.second << std::endl;
  }
  std::vector<int> result;
  for (size_t i = 0; i < config.nearest_cnt; ++i) {
    result.push_back(config.indicators[qwe[i].second]);
  }
  return result;
}

}  // namespace impl

class Executor {
public:
  Executor(std::vector<std::vector<double>> data, ExecutorConfig config):
    data_(std::move(data)), config_(std::move(config)) {}

  std::vector<bool> calculate_result() {
    std::vector<bool> answer;
    for (const auto& item : data_) {
      auto nearest_known_items = find_nearest_known_items(item, config_);
      size_t survived_cnt = 0;
      for (const auto& known_item : nearest_known_items) {
        survived_cnt += known_item;
      }
      answer.push_back(survived_cnt >= config_.need_to_survive);
    }
    return answer;
  }

private:
  std::vector<std::vector<double>> data_;
  ExecutorConfig config_;
};

}  // namespace ml_workflow
