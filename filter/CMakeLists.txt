cmake_minimum_required(VERSION 2.8.4)
project(filter)

file(GLOB SOURCES "filter.cpp")

add_subdirectory(../latte/logger logger)
add_subdirectory(../latte/command_parser command_parser)

add_library(filter ${SOURCES})
target_link_libraries(filter pthread logger command_parser)

add_executable(filter_bin "launcher.cpp")
target_link_libraries(filter_bin filter)
set_target_properties(filter_bin PROPERTIES OUTPUT_NAME filter)
