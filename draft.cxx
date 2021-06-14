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
      : _X(X), _K(_X.cols, components ? components : _X.cols),
        _W(_K.cols, _K.cols), _A(_K.cols, _K.cols), _S(_X.rows, _X.cols),
       _center(1, _X.cols) {}
  ~libICA() {}

  class mat : public cv::Mat_<double> {
  public:
    using cv::Mat_<double>::Mat_;
    using cv::Mat_<double>::operator=;

    operator double **() {
      if (rowptrs.size() != rows) {
        rowptrs.resize(rows);
        rowptrs[0] = 0;
      }
      if (rowptrs[0] != reinterpret_cast<double *>(ptr(0))) {
        for (size_t w = 0; w < rows; w++) {
          rowptrs[w] = reinterpret_cast<double *>(ptr(w));
        }
      }
      return rowptrs.data();
    }

  private:
    std::vector<double *> rowptrs;
  };

  mat const &calculate(cv::Mat_<double> const &X = {}) {
    if (X.rows) {
      if (X.cols != _X.cols) {
        throw std::logic_error("column count differs");
      }
      _X = X;
      _S.resize(_X.rows, _X.cols);
    }
    _center = 0;
    double inverse_rows = 1.0 / _X.rows;
    for (size_t col = 0; col < _X.cols; col ++) {
      _center(col) = (cv::sum(_X.col(col)) * inverse_rows)[0];
    }
    fastICA(_X, _X.rows, _X.cols, _K.cols, _K, _W, _A, _S);
    return _S;
  }

  mat &X() { return _X; }
  mat const &K() { return _K; }
  mat const &W() { return _W; }
  mat const &A() { return _A; }
  mat const &S() { return _S; }
  mat &mixed() { return X(); }
  mat const &unmixed() { return S(); }
  // unmixing and mixing assume the data is centered, i.e. DC offset subtracted
  mat const &center() { return _center; }
  cv::MatExpr centered() { return X() - cv::Mat_<double>(_X.rows, 1, 1) * center(); }
  cv::MatExpr prewhitened() { return centered() * prewhitening(); }
  mat const &prewhitened_unmixing() { return W(); }
  cv::MatExpr unmixing() { return prewhitening() * prewhitened_unmixing(); }
  mat const &prewhitening() { return K(); }
  mat const &mixing() { return A(); }

private:
  mat _X, _K, _W, _A, _S;
  // _center is really an extra DC column
  mat _center;
};

class cisseimpact_FastICA {
public:
  cisseimpact_FastICA(cv::Mat const &mixed) : _mixed(mixed) {}

  cv::Mat const &calculate(cv::Mat const &mixed = {}) {
    if (mixed.rows) {
      _mixed = mixed;
    }
    remean(_mixed, _mixed);
    whiten(_mixed, _mixed, _E, _D);
    runICA(_mixed, _unmixed, _unmixing, _mixed.cols);
    return _unmixed;
  }

  cv::Mat_<double> const & mixed() { return _mixed; }
  cv::Mat_<double> const & unmixed() { return _unmixed; }
  cv::Mat_<double> const & unmixing() { return _unmixing; }

private:
  cv::Mat_<double> _mixed, _unmixed, _unmixing, _D, _E;

  static void remean(cv::Mat input, cv::Mat &output) {
    cv::Mat mean;
    cv::reduce(input, mean, 0, CV_REDUCE_AVG);
    cv::Mat temp = cv::Mat::ones(input.rows, 1, CV_64FC1);
    output = input - temp * mean;
  }
  static void remean(cv::Mat &input, cv::Mat &output, cv::Mat &mean) {
    cv::reduce(input, mean, 0, CV_REDUCE_AVG);
    cv::Mat temp = cv::Mat::ones(input.rows, 1, CV_64FC1);
    output = input - temp * mean;
  }
  static void whiten(cv::Mat input, cv::Mat &output) {
    // need to be remean before whiten

    const int N = input.rows; // num of data
    const int M = input.cols; // dimention

    cv::Mat cov;
    cv::Mat D;
    cv::Mat E;
    cv::Mat temp = cv::Mat::eye(M, M, CV_64FC1);
    cv::Mat temp2;

    cov = input.t() * input / N;
    cv::eigen(cov, D, E);
    cv::sqrt(D, D);

    for (int i = 0; i < M; i++) {
      temp.at<double>(i, i) = D.at<double>(i, 0);
    }

    temp2 = E * temp.inv() * E.t() * input.t();

    output = temp2.t();
  }

  static void whiten(cv::Mat input, cv::Mat &output, cv::Mat &E, cv::Mat &D) {
    // need to be remean before whiten

    const int N = input.rows; // num of data
    const int M = input.cols; // dimention

    cv::Mat cov;
    cv::Mat D2;
    cv::Mat temp = cv::Mat::eye(M, M, CV_64FC1);
    cv::Mat temp2;
    cv::Mat E2;

    cov = input.t() * input / N;
    cv::eigen(cov, D, E2);
    cv::sqrt(D, D2);
    E = E2.t();

    for (int i = 0; i < M; i++) {
      temp.at<double>(i, i) = D2.at<double>(i, 0);
    }

    temp2 = E2 * temp.inv() * E2.t() * input.t();

    output = temp2.t();
  }

  static void
  runICA(cv::Mat input, cv::Mat &output, cv::Mat &W,
         int snum) // output =Independent components matrix,W=Un-mixing matrix
  {
    const int M = input.rows; // number of data
    const int N = input.cols; // data dimension

    const int maxIterations = 1000;
    const double epsilon = 0.0001;

    if (N < snum) {
      snum = M;
      printf(" Can't estimate more independent components than dimension of "
             "data ");
    }

    cv::Mat R(snum, N, CV_64FC1);
    cv::randn(R, cv::Scalar(0), cv::Scalar(1));
    cv::Mat ONE = cv::Mat::ones(M, 1, CV_64FC1);

    for (int i = 0; i < snum; ++i) {
      int iteration = 0;
      cv::Mat P(1, N, CV_64FC1);
      R.row(i).copyTo(P.row(0));

      while (iteration <= maxIterations) {
        iteration++;
        cv::Mat P2;
        P.copyTo(P2);
        cv::Mat temp1, temp2, temp3, temp4;
        temp1 = P * input.t();
        cv::pow(temp1, 3, temp2);
        cv::pow(temp1, 2, temp3);
        temp3 = 3 * temp3;
        temp4 = temp3 * ONE;
        P = temp2 * input / M - temp4 * P / M;

        if (i != 0) {
          cv::Mat temp5;
          cv::Mat wj(1, N, CV_64FC1);
          cv::Mat temp6 = cv::Mat::zeros(1, N, CV_64FC1);

          for (int j = 0; j < i; ++j) {
            R.row(j).copyTo(wj.row(0));
            temp5 = P * wj.t() * wj;
            temp6 = temp6 + temp5;
          }
          P = P - temp6;
        }
        double Pnorm = cv::norm(P, 4);
        P = P / Pnorm;

        double j1 = cv::norm(P - P2, 4);
        double j2 = cv::norm(P + P2, 4);
        if (j1 < epsilon || j2 < epsilon) {
          P.row(0).copyTo(R.row(i));
          break;
        } else if (iteration == maxIterations) {
          P.row(0).copyTo(R.row(i));
        }
      }
    }
    output = R * input.t();
    W = R;
  }
};

int main(int argc, char *const *argv) {
  cv::VideoCapture cap("MOV_20210514_1424307.mp4");

  if (!cap.isOpened()) {
    std::cerr << "Failed to open MOV_20210514_1424307.mp4" << std::endl;
    return -1;
  }

  int frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  double fps = cap.get(cv::CAP_PROP_FPS);

  libICA ica(cv::Mat_<double>(frames, 3));
  //cisseimpact_FastICA ica(cv::Mat_<double>(frames, 3));

  cv::Mat sequence, frame;
  std::cerr << "Loading frames ..." << std::endl;
  for (int framenum = 0; framenum < frames; ++framenum) {
    cap >> frame;
    cv::Mat_<double> mean(
        1, 3, cv::mean(frame).val); // take mean as a 1x3 matrix of channels
    mean.copyTo(ica.mixed().row(framenum));
    if (0 == framenum % 16 || framenum + 1 == frames) {
      std::cerr << "\r" << (framenum + 1) << " / " << frames << ": " << mean
                << std::flush;
    }
  }
  std::cerr << std::endl;

  ica.calculate();

  //std::cout << "mixing * unmixing = " << ica.mixing() * ica.unmixing()
  //          << std::endl;
  //std::cout << "mixing * T(unmixing) = " << ica.mixing() * ica.unmixing().t()
  //          << std::endl;
  //std::cout << "T(mixing) * unmixing = " << ica.mixing().t() * ica.unmixing()
  //          << std::endl;
  //std::cout << "T(mixing) * T(unmixing) = "
  //          << ica.mixing().t() * ica.unmixing().t() << std::endl;

  std::cout << "mixed, row 1 = " << ica.mixed().row(0) << std::endl;
  std::cout << "center = " << ica.center() << std::endl;
  std::cout << "mixed_centered, row 1 = " << ica.centered().row(0) << std::endl;
  //std::cout << "mixed_prewhitened, row 1 = " << ica.prewhitened().row(0) << std::endl;
  //std::cout << "mixed_centered * prewhitening, row 1 = "
  //          << (ica.centered() * ica.prewhitening()).row(0) << std::endl;
  std::cout << "unmixed, row 1 = " << ica.unmixed().row(0) << std::endl;
  //std::cout << "mixed_prewhitened * prewhitened_unmixing, row 1 = "
  //          << (ica.prewhitened() * ica.prewhitened_unmixing()).row(0) << std::endl;
  std::cout << "mixed_centered * unmixing, row 1 = "
            << (ica.centered() * ica.unmixing()).row(0) << std::endl;
  std::cout << "unmixed * mixing, row 1 = "
            << (ica.unmixed() * ica.mixing()).row(0) << std::endl;
  std::cout << "mixed_centered * prewhitening * prewhitened_unmixing, row 1 = "
            << (ica.centered() * ica.prewhitening() * ica.prewhitened_unmixing()).row(0)
            << std::endl;
  std::cout << "mixed_prewhitened * prewhitened_unmixing, row 1 = "
	    << (ica.prewhitened() * ica.prewhitened_unmixing()).row(0)
	    << std::endl;
  std::cout << "mixing * unmixing = "
            << ica.mixing() * ica.unmixing() << std::endl;

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
