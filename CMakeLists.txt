cmake_minimum_required(VERSION 3.10)

# disabling these by default lets a default build succeed in termux
set(BUILD_ANDROID_PROJECTS OFF CACHE BOOL "Build Android projects providing .apk files")
set(WITH_CPUFEATURES OFF CACHE BOOL "Use cpufeatures Android library")
add_subdirectory(opencv)
include_directories(
	${CMAKE_BINARY_DIR}
	opencv/modules/core/include
	opencv/modules/videoio/include
)

project(telemedidea)

add_executable(draft draft.cxx)
target_link_libraries(draft opencv_videoio)
