cmake_minimum_required(VERSION 3.10)

add_subdirectory(opencv)

# set the project name
project(telemedidea)

# add the executable
add_executable(Tutorial tutorial.cxx)
