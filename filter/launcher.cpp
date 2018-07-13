// Author: Evgenii Kazakov. Github: @evgenstf
#include "../latte/logger/logger.h"
#include "../latte/command_parser/command_parser.h"
#include "filter.h"
#include <iostream>
#include <vector>

using namespace latte;
using namespace ml_workflow;

static const std::unordered_set<std::string> kAvaliableCommands{"--filtered", "--data"};

int main(int arguments_count, char* arguments[]) {
  SET_LOG_LEVEL(Info);
  CommandParser parser(kAvaliableCommands);
  auto parse_result = parser.parse(arguments_count, arguments);

  std::string data;
  std::string filtered;
  for (const auto& command : parse_result) {
    if (command.first == "--filtered") {
      filtered = command.second.front();
    } else if (command.first == "--data") {
      data = command.second.front();
    }
  }

  Filter filter(data);
  filter.write_to_file(filtered);
  return 0;
}
