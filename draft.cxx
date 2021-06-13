#include <iostream>

#include <opencv2/videoio.hpp>

int main(int argc, char *const *argv)
{
  cv::VideoCapture cap("MOV_20210514_1424307.mp4");

  if (!cap.isOpened()) {
    std::cerr << "Failed to open MOV_20210514_1424307.mp4" << std::endl;
    return -1;
  }

  cv::Mat frame;
  cap >> frame;
  std::cout << "mean of frame 1: " << cv::mean(frame) << std::endl;
}
