// Author: Evgenii Kazakov. Github: @evgenstf
#include "../latte/logger/logger.h"
#include "../latte/command_parser/command_parser.h"
#include "executor.h"
#include "executor_config.h"
#include <iostream>
#include <vector>

using namespace latte;
using namespace ml_workflow;

static const std::unordered_set<std::string> kAvaliableCommands{"--config", "--data"};

auto generate_data() {
  std::vector<std::vector<double>> result;
  for (int i = 0; i < 10; ++i) {
    result.emplace_back();
    for (int j = 0; j < 4; ++j) {
      result.back().push_back(0);
    }
  }
  return result;
}

int main(int arguments_count, char* arguments[]) {
  SET_LOG_LEVEL(Info);
  CommandParser parser(kAvaliableCommands);
  auto parse_result = parser.parse(arguments_count, arguments);

  std::string data;
  std::string executor_config = "config.txt";
  for (const auto& command : parse_result) {
    if (command.first == "--config") {
      executor_config = command.second.front();
    } else if (command.first == "--data") {
      data = command.second.front();
    }
  }

  Executor executor(generate_data(), ExecutorConfig(executor_config));
  ExecutorConfig(executor_config).write_to_file("kek.json");
  auto result = executor.calculate_result();
  for (const auto& c : result) {
    std::clog << c << std::endl;
  }
  return 0;
}
