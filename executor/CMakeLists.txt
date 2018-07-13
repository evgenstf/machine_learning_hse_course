cmake_minimum_required(VERSION 2.8.4)
project(executor)

file(GLOB SOURCES "executor.cpp")

add_subdirectory(../latte/logger logger)
add_subdirectory(../latte/command_parser command_parser)

add_library(executor ${SOURCES})
target_link_libraries(executor pthread logger command_parser)

add_executable(executor_bin "launcher.cpp")
target_link_libraries(executor_bin executor)
set_target_properties(executor_bin PROPERTIES OUTPUT_NAME executor)
