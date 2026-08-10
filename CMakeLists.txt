cmake_minimum_required(VERSION 3.15)

project(MLInC C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SRC_FILES
    src/*.c
)

add_executable(mlinc ${SRC_FILES})

target_link_libraries(mlinc m)