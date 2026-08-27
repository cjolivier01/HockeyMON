#include "hockeymon/csrc/postprocess/ImagePostProcess.h"

#include <iostream>
#include <sstream>

#include <Python.h>
#include <frameobject.h>

#include "absl/strings/str_split.h"

namespace hm {

namespace {
// Function to merge two YAML nodes
YAML::Node merge_yaml_nodes(
    const YAML::Node& node1,
    const YAML::Node& update_with_node) {
  if (node1.IsMap() && update_with_node.IsMap()) {
    // Merge two maps
    YAML::Node result(YAML::NodeType::Map);
    for (const auto& it1 : node1) {
      if (update_with_node[it1.first]) {
        result[it1.first] =
            merge_yaml_nodes(it1.second, update_with_node[it1.first]);
      } else {
        result[it1.first] = it1.second;
      }
    }
    for (const auto& it2 : update_with_node) {
      if (!node1[it2.first]) {
        result[it2.first] = it2.second;
      }
    }
    return result;
  } else if (node1.IsSequence() && update_with_node.IsSequence()) {
    // Merge two sequences
    YAML::Node result(YAML::NodeType::Sequence);
    for (const auto& elem : node1) {
      result.push_back(elem);
    }
    for (const auto& elem : update_with_node) {
      result.push_back(elem);
    }
    return result;
  } else {
    // For other types (e.g., scalars), simply choose one of the nodes
    return node1.IsNull() ? update_with_node : node1;
  }
}

} // namespace

std::string HMPostprocessConfig::to_string() const {
  std::stringstream ss;
  ss << "use_watermark = " << (use_watermark ? "true" : "false") << "\n";
  return ss.str();
}

ImagePostProcessor::ImagePostProcessor(
    std::shared_ptr<HMPostprocessConfig> postprocess_config,
    std::string config_file)
    : postprocess_config_(std::move(postprocess_config)) {
  // if (!config_file.empty()) {
  //   std::vector<std::string> config_files = absl::StrSplit(config_file, ',');
  //   for (const auto& yaml_file_path : config_files) {
  //     try {
  //       YAML::Node yaml_file_node = YAML::LoadFile(yaml_file_path);
  //       config_ = merge_yaml_nodes(config_, yaml_file_node);
  //     } catch (const YAML::Exception& e) {
  //       std::stringstream ss;
  //       ss << "Error parsing YAML file: \"" << yaml_file_path
  //          << "\": " << e.what() << std::endl;
  //       throw std::runtime_error(ss.str());
  //     }
  //   }
  // }
}

} // namespace hm

namespace {
std::string get_python_string(PyObject* obj) {
  std::string str;
  PyObject* temp_bytes = PyUnicode_AsEncodedString(obj, "UTF-8", "strict");
  if (temp_bytes != NULL) {
    const char* s = PyBytes_AS_STRING(temp_bytes);
    str = s;
    Py_DECREF(temp_bytes);
  }
  return str;
}
}

extern "C" __attribute__((visibility("default"))) int __py_bt() {

  // int count = 0;
  // PyThreadState* tstate = PyThreadState_GET();
  // if (NULL != tstate && NULL != tstate->frame) {
  //   PyFrameObject* frame = tstate->frame;

  //   printf("Python stack trace:\n");
  //   while (NULL != frame) {
  //     /*
  //      frame->f_lineno will not always return the correct line number
  //      you need to call PyCode_Addr2Line().
  //     */
  //     int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
  //     PyObject* temp_bytes = PyUnicode_AsEncodedString(
  //         frame->f_code->co_filename, "UTF-8", "strict");
  //     if (temp_bytes != NULL) {
  //       auto filename = get_python_string(frame->f_code->co_filename);
  //       auto funcname = get_python_string(frame->f_code->co_name);
  //       printf("    %s(%d): %s\n", filename.c_str(), line, funcname.c_str());
  //       Py_DECREF(temp_bytes);
  //     }
  //     frame = frame->f_back;
  //     ++count;
  //   }
  // }
  // return count;
  return 0;
}
