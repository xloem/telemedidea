#include <iostream>

#include <opencv2/videoio.hpp>

extern "C" {
#include <libICA.h>
#include <matrix.h>
}

class libICA {
public:
  // mixed has a row for each time, and a column for each channel
  libICA(cv::Mat const &mixed, unsigned int components = 0)
      : _mixed(mixed),
        _prewhitening(_mixed.cols, components ? components : _mixed.cols),
        _prewhitened_unmixing(_prewhitening.cols, _prewhitening.cols),
        _unmixing(_prewhitening.cols, _prewhitening.cols),
        _mixing(_prewhitening.cols, _prewhitening.cols),
        _unmixed(_mixed.rows, _mixed.cols), _center(1, _mixed.cols) {}
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

  mat const &calculate(cv::Mat_<double> const &mixed = {}) {
    if (mixed.rows) {
      if (mixed.cols != _mixed.cols) {
        throw std::logic_error("column count differs");
      }
      _mixed = mixed;
      _unmixed.resize(_mixed.rows, _mixed.cols);
    }
    _center = 0;
    double inverse_rows = 1.0 / _mixed.rows;
    for (size_t col = 0; col < _mixed.cols; col++) {
      _center(col) = (cv::sum(_mixed.col(col)) * inverse_rows)[0];
    }
    fastICA(_mixed, _mixed.rows, _mixed.cols, _prewhitening.cols, _prewhitening,
            _prewhitened_unmixing, _mixing, _unmixed);
    _unmixing = _prewhitening * _prewhitened_unmixing;
    return _unmixed;
  }

  mat &mixed() { return _mixed; }
  mat const &unmixed() { return _unmixed; }
  // unmixing and mixing assume the data is centered, i.e. DC offset subtracted
  mat const &center() { return _center; }
  cv::MatExpr centered() {
    return _mixed - cv::Mat_<double>(_mixed.rows, 1, 1) * center();
  }
  mat const &unmixing() { return _unmixing; }
  mat const &mixing() { return _mixing; }

  mat const &prewhitening() { return _prewhitening; }
  cv::MatExpr prewhitened() { return centered() * prewhitening(); }
  mat const &prewhitened_unmixing() { return _prewhitened_unmixing; }

private:
  mat _mixed, _prewhitening, _prewhitened_unmixing, _unmixing, _mixing,
      _unmixed;
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

  cv::Mat_<double> const &mixed() { return _mixed; }
  cv::Mat_<double> const &unmixed() { return _unmixed; }
  cv::Mat_<double> const &unmixing() { return _unmixing; }

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

// this uses channel means, which are also generated for the other algorithm
cv::Mat_<double> extract_temporal_component(cv::Mat_<double> timechannels, int period)
{
  //cv::Mat_<double> periods = timechannels.reshape(0, timechannels.rows * timechannels.cols)().reshape(0, {timechannels.rows / period, period, timechannels.cols});
  int period_dims[] = {
    timechannels.rows / period,
    period,
    timechannels.cols
  };
  if (! timechannels.isContinuous()) {
    throw std::logic_error("unimplemented: noncontinuous data");
  }
  cv::Mat_<double> periods(3, period_dims, timechannels[0]);
  cv::Mat_<double> waveforms(period, timechannels.cols);
  //cv::Mat_<double> magnitudes({timechannels.cols});
  for (int w = 0; w < timechannels.cols; w ++) {
    // take mean separately for each channel dimension, removing the 1st and 2nd dimensions
    auto channel_mean = cv::mean(periods({cv::Range::all(), cv::Range::all(), {w,w+1}}));
    for (int x = 0; x < period; x ++) {
      // sum along first dimension, removing it, and subtract the calculated mean
      // makes graph of the component
      waveforms(x, w) = cv::sum(periods({cv::Range::all(), {x,x+1}, {w,w+1}}) - channel_mean)[0];
    }
    // then sum abs along the second dimension, and you have vector of 1hz component magnitude.
    //magnitudes(w) = cv::sum(cv::abs(waveforms.col(w)));
  }
  return waveforms;
}

template <typename Ica>
void verify_ica_result(Ica & ica)
{
  constexpr double EPSILON = 0.000001;
  if (cv::sum(cv::abs(ica.centered() * ica.unmixing() - ica.unmixed()))[0] > EPSILON) {
    throw std::logic_error("centered data doesn't unmix to unmixed data");
  }
  if (cv::sum(cv::abs(ica.unmixed() * ica.mixing() - ica.centered()))[0] > EPSILON) {
    throw std::logic_error("unmixed data doesn't mix to centered data");
  }
  if (cv::sum(cv::abs(ica.mixing() * ica.unmixing() - cv::Mat_<double>::eye(ica.mixing().rows, ica.mixing().cols)))[0] > EPSILON) {
    throw std::logic_error("mixing and unmixing matrices don't multiply to identity");
  }
}

int main(int argc, char *const *argv) {
  cv::VideoCapture cap("MOV_20210514_1424307.mp4");

  if (!cap.isOpened()) {
    std::cerr << "Failed to open MOV_20210514_1424307.mp4" << std::endl;
    return -1;
  }

  int frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  double fps = cap.get(cv::CAP_PROP_FPS);

  for (auto range : {cv::Range(0, frames/32/2), cv::Range(frames/32/2, frames/32)}) {
    int frames = range.end - range.start;

    libICA ica(cv::Mat_<double>(frames, 3));
    // cisseimpact_FastICA ica(cv::Mat_<double>(frames, 3));

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
    verify_ica_result(ica);
  
    std::cout << "unmixed, row 1 = " << ica.unmixed().row(0) << std::endl;
    std::cout << "mixed, row 1 = " << ica.mixed().row(0) << std::endl;
    std::cout << "unmixing = " << ica.unmixing() << std::endl;
    std::cout << "mixing = " << ica.mixing() << std::endl;
  
    cv::Mat_<double> source_persec_waveforms = extract_temporal_component(ica.unmixed(), fps);
    cv::Mat_<double> mixed_persec_waveforms = extract_temporal_component(ica.mixed(), fps);
    cv::Mat_<double> source_persec_magnitudes(1, 3);
    cv::Mat_<double> mixed_persec_magnitudes(1, 3);
    for (size_t w = 0; w < 3; w ++) {
      source_persec_magnitudes(w) = cv::sum(cv::abs(source_persec_waveforms.col(w)))[0];
      mixed_persec_magnitudes(w) = cv::sum(cv::abs(mixed_persec_waveforms.col(w)))[0];
    }
    std::cout << "60bpm unmixed magnitudes: " << source_persec_magnitudes << std::endl;
    std::cout << "60bpm mixed magnitudes: " << mixed_persec_magnitudes << std::endl;
  }

  return 0;
}
