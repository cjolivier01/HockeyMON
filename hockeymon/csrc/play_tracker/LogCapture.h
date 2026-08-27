#pragma once

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace hm {
namespace play_tracker {

enum class HmLogLevel {
  kDebug = 0,
  kInfo = 1,
  kWarning = 2,
  kError = 3,
};

struct HmLogMessage {
  HmLogLevel level{HmLogLevel::kInfo};
  std::string message;
};

struct LogCaptureState {
  std::vector<HmLogMessage>* sink{nullptr};
  bool debug_to_stdout{false};
};

inline LogCaptureState& log_capture_state() {
  static thread_local LogCaptureState state;
  return state;
}

inline bool has_log_capture_sink() {
  return log_capture_state().sink != nullptr;
}

class ScopedLogCapture {
 public:
  ScopedLogCapture(std::vector<HmLogMessage>* sink, bool debug_to_stdout)
      : prev_(log_capture_state()) {
    auto& state = log_capture_state();
    state.sink = sink;
    state.debug_to_stdout = debug_to_stdout;
  }

  ~ScopedLogCapture() {
    log_capture_state() = prev_;
  }

  ScopedLogCapture(const ScopedLogCapture&) = delete;
  ScopedLogCapture& operator=(const ScopedLogCapture&) = delete;

 private:
  LogCaptureState prev_;
};

inline void hm_log(HmLogLevel level, std::string message) {
  auto& state = log_capture_state();
  if (state.sink != nullptr) {
    state.sink->push_back(HmLogMessage{level, message});
  }
  if (state.debug_to_stdout) {
    std::fprintf(stdout, "%s\n", message.c_str());
    std::fflush(stdout);
  }
}

inline void hm_log_debug(std::string message) {
  hm_log(HmLogLevel::kDebug, std::move(message));
}

inline void hm_log_info(std::string message) {
  hm_log(HmLogLevel::kInfo, std::move(message));
}

inline void hm_log_warning(std::string message) {
  hm_log(HmLogLevel::kWarning, std::move(message));
}

inline void hm_log_error(std::string message) {
  hm_log(HmLogLevel::kError, std::move(message));
}

} // namespace play_tracker
} // namespace hm

