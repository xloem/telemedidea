#include <iostream>

#include <opencv2/videoio.hpp>

extern "C" {
#include <libICA.h>
#include <matrix.h>
}

class libICA {
public:
  // X has a row for each time, and a column for each channel
  libICA(cv::Mat const &X, unsigned int compc = 0)
      : reserved_rows(0), cols(X.cols),
        compc(compc && compc <= cols ? compc : cols), _X(0),
        _K(mat_create(cols, compc)), _W(mat_create(compc, compc)),
        _A(mat_create(compc, compc)), _S(0) {
    setX(X);
  }
  ~libICA() {
    if (0 != _S) {
      mat_delete(_S, reserved_rows, cols);
    }
    mat_delete(_A, compc, compc);
    mat_delete(_W, compc, compc);
    mat_delete(_K, cols, compc);
    if (0 != _X) {
      mat_delete(_X, reserved_rows, cols);
    }
  }

  const cv::Mat_<double> X() { return {rows, cols, _X[0]}; }
  const cv::Mat_<double> K() { return {cols, compc, _K[0]}; }
  const cv::Mat_<double> W() { return {compc, compc, _W[0]}; }
  const cv::Mat_<double> A() { return {compc, compc, _A[0]}; }
  const cv::Mat_<double> S() { return {rows, cols, _S[0]}; }

  void setX(cv::Mat_<double> const &X) {
    if (X.cols != cols) {
      throw std::logic_error("column count differs");
    }
    if (X.rows > reserved_rows) {
      if (_X) {
        mat_delete(_X, reserved_rows, cols);
      }
      if (_S) {
        mat_delete(_S, reserved_rows, cols);
      }
      reserved_rows = X.rows;
      _X = mat_create(reserved_rows, cols);
      _S = mat_create(reserved_rows, cols);
    }
    rows = X.rows;
    cv::Mat_<double> const __X = this->X();
    const_cast<cv::Mat_<double> &>(__X) = X;
  }

  cv::Mat_<double> calculate(cv::Mat_<double> const &X = {}) {
    if (X.rows) {
      setX(X);
    }
    fastICA(_X, rows, cols, compc, _K, _W, _A, _S);
    return S();
  }

private:
  int reserved_rows, rows, cols, compc;
  mat _X; // contiguous rows x cols // input mixed data, rows = time
  mat _K; // contiguous compc x cols
  mat _W; // contiguous compc x compc
  mat _A; // contiguous compc x compc
  mat _S; // contiguous rows x cols // unmixed data, rows = time
};

int main(int argc, char *const *argv) {
  cv::VideoCapture cap("MOV_20210514_1424307.mp4");

  if (!cap.isOpened()) {
    std::cerr << "Failed to open MOV_20210514_1424307.mp4" << std::endl;
    return -1;
  }

  libICA ica(cv::Mat_<double>(cap.get(cv::CAP_PROP_FRAME_COUNT), 3));

  size_t frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  cv::Mat sequence, frame;
  std::cerr << "Loading frames ..." << std::endl;
  for (size_t framenum = 0; framenum < frames; ++framenum) {
    cap >> frame;
    auto mean = cv::mean(frame);
    ica.X().row(framenum) = cv::Mat_<double>(mean, false);
    if (0 == framenum % 16) {
      std::cerr << (framenum * 100) / frames << "%: " << mean << "\r"
                << std::flush;
    }
  }

  return 0;
}
