#include <iostream>

#include <opencv2/videoio.hpp>

extern "C" {
#include <libICA.h>
#include <matrix.h>
}

class libICA {
public:
  // X has a row for each time, and a column for each channel
  libICA(cv::Mat const &X, unsigned int components = 0)
      : _X(X),
        _K(_X.cols, components ? components : _X.cols),
        _W(_K.cols, _K.cols),
        _A(_K.cols, _K.cols),
        _S(_X.rows, _X.cols) { }
  ~libICA() { }

  class mat : public cv::Mat_<double>
  {
  public:
    using cv::Mat_<double>::Mat_;

    operator double**() {
      if (rowptrs.size() != rows) {
        rowptrs.resize(rows);
        rowptrs[0] = 0;
      }
      if (rowptrs[0] != reinterpret_cast<double *>(ptr(0))) {
        for (size_t w = 0; w < rows; w ++) {
          rowptrs[w] = reinterpret_cast<double *>(ptr(w));
        }
      }
      return rowptrs.data();
    }
  private:
    std::vector<double *> rowptrs;
  };

  mat const & calculate(cv::Mat_<double> const &X = {}) {
    if (X.rows) {
      if (X.cols != _X.cols) {
        throw std::logic_error("column count differs");
      }
      _X = X;
      _S.resize(_X.rows, _X.cols);
    }
    fastICA(_X, _X.rows, _X.cols, _K.cols, _K, _W, _A, _S);
    return _S;
  }

  mat       & X() { return _X; }
  mat const & K() { return _K; }
  mat const & W() { return _W; }
  mat const & A() { return _A; }
  mat const & S() { return _S; }
  mat       & mixed() { return X(); }
  mat const & unmixed() { return S(); }
  mat const & mixing() { return A(); }
  mat const & unmixing() { return W(); }
  mat const & prewhitening() { return K(); }

private:
  mat _X, _K, _W, _A, _S;
};

int main(int argc, char *const *argv) {
  cv::VideoCapture cap("MOV_20210514_1424307.mp4");

  if (!cap.isOpened()) {
    std::cerr << "Failed to open MOV_20210514_1424307.mp4" << std::endl;
    return -1;
  }

  libICA ica(cv::Mat_<double>(cap.get(cv::CAP_PROP_FRAME_COUNT), 3));

  int frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  double fps = cap.get(cv::CAP_PROP_FPS);
  cv::Mat sequence, frame;
  std::cerr << "Loading frames ..." << std::endl;
  for (int framenum = 0; framenum < frames; ++framenum) {
    cap >> frame;
    cv::Mat_<double> mean(1, 3, cv::mean(frame).val); // take mean as a 1x3 matrix of channels
    mean.copyTo(ica.X().row(framenum));
    if (0 == framenum % 16 || framenum + 1 == frames) {
      std::cerr << "\r" << (framenum+1) << " / " << frames << ": " << mean
                << std::flush;
    }
  }
  std::cerr << std::endl;

  ica.calculate();


  std::cout << "mixing * unmixing = " << ica.mixing() * ica.unmixing() << std::endl;
  std::cout << "mixing * T(unmixing) = " << ica.mixing() * ica.unmixing().t() << std::endl;
  std::cout << "T(mixing) * unmixing = " << ica.mixing().t() * ica.unmixing() << std::endl;
  std::cout << "T(mixing) * T(unmixing) = " << ica.mixing().t() * ica.unmixing().t() << std::endl;

  std::cout << "mixed, row 1 = " << ica.mixed().row(0) << std::endl;
  std::cout << "mixed * prewhitening, row 1 = " << (ica.mixed() * ica.prewhitening()).row(0) << std::endl;
  std::cout << "unmixed, row 1 = " << ica.unmixed().row(0) << std::endl;
  std::cout << "mixed * unmixing, row 1 = " << (ica.mixed() * ica.unmixing()).row(0) << std::endl;
  std::cout << "unmixed * mixing, row 1 = " << (ica.unmixed() * ica.mixing()).row(0) << std::endl;
  std::cout << "mixed * prewhitening * unmixing, row 1 = " << (ica.mixed() * ica.prewhitening() * ica.unmixing()).row(0) << std::endl;

  /* the opencv matrix multiplication gets the same results as the libica one
  libICA::mat result(ica.mixing().rows, ica.unmixing().cols);
  mat_mult(ica.mixing(), ica.mixing().rows, ica.mixing().cols,
           ica.unmixing(), ica.unmixing().rows, ica.unmixing().cols,
           result);
  std::cout << "mixing * unmixing = " << std::endl;
  for (size_t w = 0; w < result.rows; w ++) {
    for (size_t x = 0; x < result.cols; x ++) {
      std::cout << "\t" << result[w][x];
    }
    std::cout << std::endl;
  }
  */


  return 0;
}
