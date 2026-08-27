#include "hockeymon/csrc/common/Gpu.h"

#include "absl/debugging/failure_signal_handler.h"
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"

#include <unistd.h>

#include <GL/glew.h>
#include <GL/glx.h>
#include <X11/Xlib.h> // Include Xlib.h for XInitThreads

#include <iostream>
#include <mutex>

namespace hm {

namespace {

std::string get_executable_path() {
  char result[PATH_MAX * 2 + 1];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX * 2);
  std::string path = std::string(result, (count > 0) ? count : 0);
  return path;
}

std::mutex init_mu_;
std::size_t global_initialized{false};
thread_local bool attempted = false;
thread_local bool attempted_result = false;
} // namespace

std::size_t get_tick_count_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

void set_thread_name(const std::string& thread_name, int index) {
  if (index >= 0) {
    std::string n = thread_name;
    n += '-';
    n += std::to_string(index);
    pthread_setname_np(pthread_self(), n.c_str());
  } else {
    pthread_setname_np(pthread_self(), thread_name.c_str());
  }
}

void init_stack_trace() {
  absl::InitializeSymbolizer(get_executable_path().c_str());

  // Install the failure signal handler. This should capture various failure
  // signals (like segmentation faults) and print a stack trace.
  // absl::FailureSignalHandlerOptions options;
  // absl::InstallFailureSignalHandler(options);
}

#if 0
bool check_cuda_opengl() {
  //return false;
  if (attempted) {
    return attempted_result;
  }

  std::unique_lock<std::mutex> lk(init_mu_);
  if (!global_initialized) {
    assert(XInitThreads());
    global_initialized = true;
  }

  attempted = true;

  int argc = 2;
  char* argv[2] = {const_cast<char*>("python"), const_cast<char*>("-g")};
  attempted_result = hugin_utils::initGPU(&argc, &argv[0]);
  return attempted_result;
}

#else

static thread_local std::unique_ptr<GpuContext> gpu_context;
bool check_cuda_opengl() {
  if (!gpu_context) {
    gpu_context = std::make_unique<GpuContext>();
  }
  return gpu_context->is_valid();
}

#endif

struct ContextSettings {
  Display* display;
  XVisualInfo* visualInfo;
  GLXContext context;
  Window window;
  Colormap colormap;

  ContextSettings() {
    display = NULL;
    visualInfo = NULL;
    context = NULL;
    window = 0;
    colormap = 0;
  };
};

static thread_local ContextSettings context;
static thread_local bool valid_context = false;

bool CreateContext() {
  /* open display */
  context.display = XOpenDisplay(NULL);
  if (context.display == NULL) {
    return false;
  };
  /* query for glx */
  int erb, evb;
  if (!glXQueryExtension(context.display, &erb, &evb)) {
    return false;
  };
  /* choose visual */
  int attrib[] = {GLX_RGBA, None};
  context.visualInfo =
      glXChooseVisual(context.display, DefaultScreen(context.display), attrib);
  if (context.visualInfo == NULL) {
    return false;
  };
  /* create context */
  context.context =
      glXCreateContext(context.display, context.visualInfo, None, True);
  if (context.context == NULL) {
    return false;
  };
  /* create window */
  context.colormap = XCreateColormap(
      context.display,
      RootWindow(context.display, context.visualInfo->screen),
      context.visualInfo->visual,
      AllocNone);
  XSetWindowAttributes swa;
  swa.border_pixel = 0;
  swa.colormap = context.colormap;
  context.window = XCreateWindow(
      context.display,
      RootWindow(context.display, context.visualInfo->screen),
      0,
      0,
      1,
      1,
      0,
      context.visualInfo->depth,
      InputOutput,
      context.visualInfo->visual,
      CWBorderPixel | CWColormap,
      &swa);
  /* make context current */
  if (!glXMakeCurrent(context.display, context.window, context.context)) {
    return false;
  };
  return true;
};

void DestroyContext() {
  if (context.display != NULL && context.context != NULL) {
    glXDestroyContext(context.display, context.context);
  }
  if (context.display != NULL && context.window != 0) {
    XDestroyWindow(context.display, context.window);
  };
  if (context.display != NULL && context.colormap != 0) {
    XFreeColormap(context.display, context.colormap);
  };
  if (context.visualInfo != NULL) {
    XFree(context.visualInfo);
  };
  if (context.display != NULL) {
    XCloseDisplay(context.display);
  };
};

void maybe_set_display(int disp_nr) {
  const char* s = getenv("DISPLAY");
  if (s && *s) {
    return;
  }
  std::string disp = std::string(":") + std::to_string(disp_nr);
  setenv("DISPLAY", disp.c_str(), true);
  std::cout << "Set DISPLAY to: \"" << disp << "\"";
}

GpuContext::GpuContext() {
  // currently, ripper is display 1: for some reason
  maybe_set_display(1);
  valid_context = CreateContext();
  if (valid_context) {
    std::cout << "HockeyMON using graphics card: " << glGetString(GL_VENDOR)
              << " " << glGetString(GL_RENDERER) << std::endl;

    int err = glewInit();
    if (err != GLEW_OK) {
      std::cerr << "An error occurred while setting up the GPU:" << std::endl;
      std::cerr << glewGetErrorString(err) << std::endl;
      std::cerr << "Switching to CPU calculation." << std::endl;
      DestroyContext();
      valid_context = false;
    }
  }
}

GpuContext::~GpuContext() {
  if (valid_context) {
    valid_context = false;
    DestroyContext();
  }
}

bool GpuContext::is_valid() const {
  return valid_context;
}

} // namespace hm
