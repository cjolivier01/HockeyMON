#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "hockeymon/csrc/stitcher/HomographyMaps.h"

namespace hm::stitcher {

inline void bind_akaze_features(pybind11::module_& module) {
  namespace py = pybind11;
  module.def(
      "detect_akaze_features",
      [](const py::array_t<uint8_t, py::array::c_style>& gray,
         const py::array_t<uint8_t, py::array::c_style>& mask) {
        if (gray.ndim() != 2 || mask.ndim() != 2 ||
            gray.shape(0) != mask.shape(0) || gray.shape(1) != mask.shape(1) ||
            gray.shape(0) > 1920 || gray.shape(1) > 1920) {
          throw std::invalid_argument(
              "AKAZE image and mask must have matching 2D shapes at most 1920 pixels per axis");
        }
        AkazeFeatures result;
        {
          py::gil_scoped_release release;
          result = detect_akaze_features(
              gray.data(), mask.data(), gray.shape(1), gray.shape(0));
        }
        py::array_t<float> points({static_cast<int>(result.points.size()), 2});
        for (size_t index = 0; index < result.points.size(); ++index) {
          points.mutable_data()[2 * index] = result.points[index][0];
          points.mutable_data()[2 * index + 1] = result.points[index][1];
        }
        py::array_t<uint8_t> descriptors(
            {static_cast<int>(result.points.size()), result.descriptor_size});
        std::copy(
            result.descriptors.begin(),
            result.descriptors.end(),
            descriptors.mutable_data());
        return py::make_tuple(points, descriptors);
      },
      py::arg("gray").noconvert(),
      py::arg("mask").noconvert());
}

} // namespace hm::stitcher
