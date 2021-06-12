cmake_minimum_required(VERSION 3.10)

# disabling this by default lets a default build succeed in termux
set(BUILD_ANDROID_PROJECTS OFF CACHE BOOL "Build Android projects providing .apk files")
add_subdirectory(opencv)

# set the project name
project(telemedidea)

# add the executable
#add_executable(Tutorial tutorial.cxx)
