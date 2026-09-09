#include "hockeymon/csrc/stitcher/AkazeBindings.h"

PYBIND11_MODULE(_akaze_test_native, module) {
  hm::stitcher::bind_akaze_features(module);
}
