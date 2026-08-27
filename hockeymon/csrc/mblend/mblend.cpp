/*
        Multiblend 2.0 (c) 2021 David Horman

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program. If not, see <https://www.gnu.org/licenses/>.

        The author can be contacted at davidhorman47@gmail.com
*/

#define NOMINMAX
#include <stdio.h>
#include <cstdint>

#include <algorithm>
#include <cassert>
#include <memory>
#include <thread>
#include <vector>

#include <Eigen/CXX11/ThreadPool>

#ifdef __APPLE__
#define memalign(a, b) malloc((b))
#else
#include <malloc.h>
#endif

#include "jpeglib.h"
#include "tiffio.h"

#ifndef _WIN32
#include <strings.h>
int _stricmp(const char* a, const char* b) {
  return strcasecmp(a, b);
}
#define ZeroMemory(a, b) memset(a, 0, b)
#define sprintf_s sprintf
#define sscanf_s sscanf
void* _aligned_malloc(size_t size, int boundary) {
  return memalign(boundary, size);
}
void _aligned_free(void* a) {
  free(a);
}
void fopen_s(FILE** f, const char* filename, const char* mode) {
  *f = fopen(filename, mode);
}
#endif

int verbosity = 1;

#include "hockeymon/csrc/mblend/mblend.h"

/* clang-format off */
#include "pnger.cpp"
#include "mapalloc.cpp"
#include "pyramid.cpp"
#include "functions.cpp"

#include "threadpool.cpp"
#include "geotiff.cpp"

namespace hm {

class PyramidWithMasks : public Pyramid {
public:
  using Pyramid::Pyramid;
  std::vector<std::shared_ptr<Flex>> masks;
  ~PyramidWithMasks() {
    // for (auto m : masks) {
    //   delete m;
    // }
  }
};

enum class ImageType { MB_NONE, MB_TIFF, MB_JPEG, MB_PNG, MB_MEM };
}

#include "image.cpp"
/* clang-format on */

#ifdef _WIN32
FILE _iob[] = {*stdin, *stdout, *stderr};

extern "C" FILE* __cdecl __iob_func(void) {
  return _iob;
}
#pragma comment(lib, "legacy_stdio_definitions.lib")
// the above are required to support VS 2010 build of libjpeg-turbo 2.0.6
#pragma comment(lib, "tiff.lib")
#pragma comment(lib, "turbojpeg-static.lib")
#pragma comment(lib, "libpng16.lib")
#pragma comment(lib, "zlib.lib")
#endif

#define MASKVAL(X) \
  (((X)&0x7fffffffffffffff) | image_state.images[(X)&0xffffffff]->mask_state)

namespace hm {

void BlenderImageState::init_from_images(
    std::vector<std::reference_wrapper<hm::MatrixRGB>> incoming_images) {
  images.clear();
  images.reserve(incoming_images.size());
  // TODO: do incoming_images -> image_state.images ouside (maybe member func to
  // BlenderImageState)
  for (auto& img : incoming_images) {
    std::size_t size =
        img.get().rows() * img.get().cols() * img.get().channels();
    std::vector<size_t> shape{
        img.get().rows(), img.get().cols(), img.get().channels()};
    images.emplace_back(std::make_unique<Image>(
        img.get().data(), size, std::move(shape), img.get().xy_pos()));
  }
}

void BlenderImageState::init_from_image_state(
    const BlenderImageState& prev_image_state,
    const std::vector<std::reference_wrapper<hm::MatrixRGB>>& incoming_images) {
  images.clear();
  images.reserve(incoming_images.size());
  assert(prev_image_state.images.size() == incoming_images.size());
  for (std::size_t i = 0, n = incoming_images.size(); i < n; ++i) {
    auto& prev_image = prev_image_state.images.at(i);
    auto new_image = prev_image->clone_with_new_data(
        incoming_images.at(i).get().data(), /*own=*/false);
    images.emplace_back(std::make_unique<Image>(std::move(new_image)));
  }
}

class Blender {
 private:
  /***********************************************************************
   * Variables
   ***********************************************************************/
  // std::vector<std::unique_ptr<Image>> images;
  int fixed_levels = 0;
  int add_levels = 0;

  int width = 0;
  int height = 0;

  uint64_t total_pixels = 0;
  uint64_t channel_totals[3] = {0};

  std::shared_ptr<Flex> full_mask_ptr_;
  std::shared_ptr<Flex> xor_mask_ptr_;
  int min_xpos = 0x7fffffff;
  int min_ypos = 0x7fffffff;

  bool no_mask = false;
  bool big_tiff = false;
  bool bgr = false;
  bool wideblend = false;
  bool reverse = false;
  bool timing = false;
  bool dither = true;
  bool gamma = false;
  bool all_threads = false;
  int wrap = 0;

  TIFF* tiff_file = NULL;
  FILE* jpeg_file = NULL;
  // Pnger* png_file = NULL;
  //  std::unique_ptr<Image> output_image_ptr_;
  ImageType output_type = ImageType::MB_NONE;
  int jpeg_quality = -1;
  int compression = -1;
  std::string seamsave_filename;
  char* seamload_filename = NULL;
  std::string xor_filename;
  char* output_filename = NULL;
  int output_bpp = 0;

  double images_time = 0;
  double copy_time = 0;
  double seam_time = 0;
  double shrink_mask_time = 0;
  double shrink_time = 0;
  double laplace_time = 0;
  double blend_time = 0;
  double collapse_time = 0;
  double wrap_time = 0;
  double out_time = 0;
  double write_time = 0;

  // std::size_t pass = 0;

  Timer timer_all, timer;

  int blend_levels = 0;
  // ThreadPool* threadpool = nullptr;
  int total_levels = 0;

  std::vector<std::shared_ptr<PyramidWithMasks>> wrap_pyramids;
  int wrap_levels_h = 0;
  int wrap_levels_v = 0;

  // std::unique_ptr<Pyramid> output_pyramid;

  static absl::Mutex thread_pool_mu_;
  static std::unique_ptr<Eigen::ThreadPool> thread_pool_;
  static std::size_t instance_count_;

  /**
   * @brief State data to reproduce a particular blend context
   */
  struct BlenderState {};

 public:
  Blender();
  ~Blender();

  // Like a null check
  static inline bool is(const std::string& s) {
    return !s.empty();
  }

  static Eigen::ThreadPool* gtp() {
    return thread_pool_.get();
  }

  int multiblend_main(int argc, char* argv[], BlenderImageState& image_state) {
    // This is here because of a weird problem encountered during development
    // with Visual Studio. It should never be triggered.
    if (verbosity != 1) {
      printf("bad compile?\n");
      exit(EXIT_FAILURE);
    }

    int i;
    // Timer timer_all, timer;
    timer_all.Start();

    TIFFSetWarningHandler(NULL);

    /***********************************************************************
     * Help
     ***********************************************************************/
    if (argc == 1 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") ||
        !strcmp(argv[1], "/?")) {
      Output(1, "\n");
      Output(
          1,
          "Multiblend v2.0.0 (c) 2021 David Horman        "
          "http://horman.net/mblend/\n");
      Output(
          1,
          "----------------------------------------------------------------"
          "------------\n");

      printf(
          "Usage: mblend [options] [-o OUTPUT] INPUT [X,Y] [INPUT] [X,Y] "
          "[INPUT]...\n");
      printf("\n");
      printf("Options:\n");
      printf(
          "  --levels X / -l X      X: set number of blending levels to X\n");
      printf(
          "                        -X: decrease number of blending levels by "
          "X\n");
      printf(
          "                        +X: increase number of blending levels by "
          "X\n");
      printf(
          "  --depth D / -d D       Override automatic output image depth (8 "
          "or 16)\n");
      printf("  --bgr                  Swap RGB order\n");
      printf(
          "  --wideblend            Calculate number of levels based on "
          "output image size,\n");
      printf("                         rather than input image size\n");
      printf(
          "  -w, --wrap=[mode]      Blend around images boundaries (NONE "
          "(default),\n");
      printf(
          "                         HORIZONTAL, VERTICAL). When specified "
          "without a mode,\n");
      printf("                         defaults to HORIZONTAL.\n");
      printf(
          "  --compression=X        Output file compression. For TIFF output, "
          "X may be:\n");
      printf("                         NONE (default), PACKBITS, or LZW\n");
      printf(
          "                         For JPEG output, X is JPEG quality "
          "(0-100, default 75)\n");
      printf(
          "                         For PNG output, X is PNG filter (0-9, "
          "default 3)\n");
      printf(
          "  --cache-threshold=     Allocate memory beyond X "
          "bytes/[K]ilobytes/\n");
      printf("      X[K/M/G]           [M]egabytes/[G]igabytes to disk\n");
      printf("  --no-dither            Disable dithering\n");
      printf(
          "  --tempdir <dir>        Specify temporary directory (default: "
          "system temp)\n");
      printf(
          "  --save-seams <file>    Save seams to PNG file for external "
          "editing\n");
      printf("  --load-seams <file>    Load seams from PNG file\n");
      printf(
          "  --no-output            Do not blend (for use with "
          "--save-seams)\n");
      printf(
          "                         Must be specified as last option before "
          "input images\n");
      printf("  --bigtiff              BigTIFF output\n");
      printf(
          "  --reverse              Reverse image priority (last=highest) for "
          "resolving\n");
      printf("                         indeterminate pixels\n");
      printf("  --quiet                Suppress output (except warnings)\n");
      printf("  --all-threads          Use all available CPU threads\n");
      printf(
          "  [X,Y]                  Optional position adjustment for previous "
          "input image\n");
      exit(EXIT_SUCCESS);
    }

    /***********************************************************************
    ************************************************************************
    * Parse arguments
    ************************************************************************
    ***********************************************************************/
    std::vector<char*> my_argv;

    bool skip = false;

    for (i = 1; i < argc; ++i) {
      my_argv.push_back(argv[i]);

      if (!skip) {
        int c = 0;

        while (argv[i][c]) {
          if (argv[i][c] == '=') {
            argv[i][c++] = 0;
            if (argv[i][c]) {
              my_argv.push_back(&argv[i][c]);
            }
            break;
          }
          ++c;
        }

        if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output")) {
          skip = true;
        }
      }
    }

    ThreadPool threadpool(ThreadPool::get_base_thread_pool());

    // if ((int)my_argv.size() < 3 && !output_image && incoming_images.empty())
    //   die("Error: Not enough arguments (try -h for help)");

    for (i = 0; i < (int)my_argv.size(); ++i) {
      if (!strcmp(my_argv[i], "-d") || !strcmp(my_argv[i], "--d") ||
          !strcmp(my_argv[i], "--depth") || !strcmp(my_argv[i], "--bpp")) {
        if (++i < (int)my_argv.size()) {
          output_bpp = atoi(my_argv[i]);
          if (output_bpp != 8 && output_bpp != 16) {
            die("Error: Invalid output depth specified");
          }
        } else {
          die("Error: Missing parameter value");
        }
      } else if (!strcmp(my_argv[i], "-l") || !strcmp(my_argv[i], "--levels")) {
        if (++i < (int)my_argv.size()) {
          int n;
          if (my_argv[i][0] == '+' || my_argv[i][0] == '-') {
            sscanf_s(my_argv[i], "%d%n", &add_levels, &n);
          } else {
            sscanf_s(my_argv[i], "%d%n", &fixed_levels, &n);
            if (fixed_levels == 0)
              fixed_levels = 1;
          }
          if (my_argv[i][n])
            die("Error: Bad --levels parameter");
        } else {
          die("Error: Missing parameter value");
        }
      } else if (!strcmp(my_argv[i], "--wrap") || !strcmp(my_argv[i], "-w")) {
        if (i + 1 >= (int)my_argv.size()) {
          die("Error: Missing parameters");
        }
        if (!strcmp(my_argv[i + 1], "none") || !strcmp(my_argv[i + 1], "open"))
          ++i;
        else if (
            !strcmp(my_argv[i + 1], "horizontal") ||
            !strcmp(my_argv[i + 1], "h")) {
          wrap = 1;
          ++i;
        } else if (
            !strcmp(my_argv[i + 1], "vertical") ||
            !strcmp(my_argv[i + 1], "v")) {
          wrap = 2;
          ++i;
        } else if (
            !strcmp(my_argv[i + 1], "both") || !strcmp(my_argv[i + 1], "hv")) {
          wrap = 3;
          ++i;
        } else
          wrap = 1;
      } else if (!strcmp(my_argv[i], "--cache-threshold")) {
        if (i + 1 >= (int)my_argv.size()) {
          die("Error: Missing parameters");
        }
        ++i;
        int shift = 0;
        int n = 0;
        size_t len = strlen(my_argv[i]);
        size_t threshold;
        sscanf_s(my_argv[i], "%zu%n", &threshold, &n);
        if (n != len) {
          if (n == len - 1) {
            switch (my_argv[i][len - 1]) {
              case 'k':
              case 'K':
                shift = 10;
                break;
              case 'm':
              case 'M':
                shift = 20;
                break;
              case 'g':
              case 'G':
                shift = 30;
                break;
              default:
                die("Error: Bad --cache-threshold parameter");
            }
            threshold <<= shift;
          } else {
            die("Error: Bad --cache-threshold parameter");
          }
        }
        MapAlloc::CacheThreshold(threshold);
      } else if (
          !strcmp(my_argv[i], "--nomask") || !strcmp(my_argv[i], "--no-mask"))
        no_mask = true;
      else if (
          !strcmp(my_argv[i], "--timing") || !strcmp(my_argv[i], "--timings"))
        timing = true;
      else if (!strcmp(my_argv[i], "--bigtiff"))
        big_tiff = true;
      else if (!strcmp(my_argv[i], "--bgr"))
        bgr = true;
      else if (!strcmp(my_argv[i], "--wideblend"))
        wideblend = true;
      else if (!strcmp(my_argv[i], "--reverse"))
        reverse = true;
      else if (!strcmp(my_argv[i], "--gamma"))
        gamma = true;
      else if (
          !strcmp(my_argv[i], "--no-dither") ||
          !strcmp(my_argv[i], "--nodither"))
        dither = false;
      //    else if (!strcmp(my_argv[i], "--force")) force_coverage
      //=
      // true;
      else if (!strncmp(my_argv[i], "-f", 2))
        Output(0, "ignoring Enblend option -f\n");
      else if (!strcmp(my_argv[i], "-a"))
        Output(0, "ignoring Enblend option -a\n");
      else if (!strcmp(my_argv[i], "--no-ciecam"))
        Output(0, "ignoring Enblend option --no-ciecam\n");
      else if (!strcmp(my_argv[i], "--primary-seam-generator")) {
        Output(0, "ignoring Enblend option --primary-seam-generator\n");
        ++i;
      }

      else if (!strcmp(my_argv[i], "--compression")) {
        if (++i < (int)my_argv.size()) {
          if (strcmp(my_argv[i], "0") == 0)
            jpeg_quality = 0;
          else if (atoi(my_argv[i]) > 0)
            jpeg_quality = atoi(my_argv[i]);
          else if (_stricmp(my_argv[i], "lzw") == 0)
            compression = COMPRESSION_LZW;
          else if (_stricmp(my_argv[i], "packbits") == 0)
            compression = COMPRESSION_PACKBITS;
          //        else if (_stricmp(my_argv[i], "deflate")
          //== 0) compression = COMPRESSION_DEFLATE;
          else if (_stricmp(my_argv[i], "none") == 0)
            compression = COMPRESSION_NONE;
          else
            die("Error: Unknown compression codec %s", my_argv[i]);
        } else {
          die("Error: Missing parameter value");
        }
      } else if (!strcmp(my_argv[i], "-v") || !strcmp(my_argv[i], "--verbose"))
        ++verbosity;
      else if (!strcmp(my_argv[i], "-q") || !strcmp(my_argv[i], "--quiet"))
        --verbosity;
      else if (
          (!strcmp(my_argv[i], "--saveseams") ||
           !strcmp(my_argv[i], "--save-seams")) &&
          i < (int)my_argv.size() - 1)
        seamsave_filename = my_argv[++i];
      else if (
          (!strcmp(my_argv[i], "--loadseams") ||
           !strcmp(my_argv[i], "--load-seams")) &&
          i < (int)my_argv.size() - 1)
        seamload_filename = my_argv[++i];
      else if (
          (!strcmp(my_argv[i], "--savexor") ||
           !strcmp(my_argv[i], "--save-xor")) &&
          i < (int)my_argv.size() - 1)
        xor_filename = my_argv[++i];
      else if (
          !strcmp(my_argv[i], "--tempdir") ||
          !strcmp(my_argv[i], "--tmpdir") && i < (int)my_argv.size() - 1)
        MapAlloc::SetTmpdir(my_argv[++i]);
      else if (!strcmp(my_argv[i], "--all-threads"))
        all_threads = true;
      else if (!strcmp(my_argv[i], "-o") || !strcmp(my_argv[i], "--output")) {
        if (++i < (int)my_argv.size()) {
          output_filename = my_argv[i];
          if (!*output_filename) {
            output_type = ImageType::MB_MEM;
          } else {
            char* ext = strrchr(output_filename, '.');

            if (!ext) {
              die("Error: Unknown output filetype");
            }

            ++ext;
            if (!(_stricmp(ext, "jpg") && _stricmp(ext, "jpeg"))) {
              output_type = ImageType::MB_JPEG;
              if (jpeg_quality == -1)
                jpeg_quality = 75;
            } else if (!(_stricmp(ext, "tif") && _stricmp(ext, "tiff"))) {
              output_type = ImageType::MB_TIFF;
            } else if (!_stricmp(ext, "png")) {
              output_type = ImageType::MB_PNG;
            } else {
              die("Error: Unknown file extension");
            }

            ++i;
          }
          break;
        }
      } else if (!strcmp(my_argv[i], "--no-output")) {
        ++i;
        output_type = ImageType::MB_MEM;
        break;
      } else {
        die("Error: Unknown argument \"%s\"", my_argv[i]);
      }
    }

    if (compression != -1) {
      if (output_type != ImageType::MB_TIFF) {
        Output(
            0, "Warning: non-TIFF output; ignoring TIFF compression setting\n");
      }
    } else if (output_type == ImageType::MB_TIFF) {
      compression = COMPRESSION_LZW;
    }

    if (jpeg_quality != -1 && output_type != ImageType::MB_JPEG &&
        output_type != ImageType::MB_PNG) {
      Output(
          0,
          "Warning: non-JPEG/PNG output; ignoring compression quality "
          "setting\n");
    }

    if ((jpeg_quality < -1 || jpeg_quality > 9) &&
        output_type == ImageType::MB_PNG) {
      die("Error: Bad PNG compression quality setting\n");
    }

    if (output_type == ImageType::MB_NONE && seamsave_filename.empty())
      die("Error: No output file specified");
    if (seamload_filename && !seamsave_filename.empty())
      die("Error: Cannot load and save seams at the same time");
    if (wrap == 3)
      die("Error: Wrapping in both directions is not currently supported");

    if (i < my_argv.size()) {
      if (!strncmp(my_argv[i], "--", 2))
        ++i;
    }

    /***********************************************************************
     * Push remaining arguments to images vector
     ***********************************************************************/
    int x, y, n;

    while (i < (int)my_argv.size()) {
      if (image_state.images.size()) {
        n = 0;
        sscanf_s(my_argv[i], "%d,%d%n", &x, &y, &n);
        if (!my_argv[i][n]) {
          image_state.images.back()->xpos_add = x;
          image_state.images.back()->ypos_add = y;
          i++;
          continue;
        }
      }
      image_state.images.push_back(std::make_unique<Image>(my_argv[i++]));
    }
    // threadpool = ThreadPool::GetInstance(
    //     all_threads ? 0 : std::thread::hardware_concurrency() / 2);
    return EXIT_SUCCESS;
  }

  int process_images(BlenderImageState& image_state) {
    int n_images = (int)image_state.images.size();

    if (n_images == 0)
      die("Error: No input files specified");
    if (!seamsave_filename.empty() && n_images > 256) {
      seamsave_filename.clear();
      Output(0, "Warning: seam saving not possible with more than 256 images");
    }
    if (seamload_filename && n_images > 256) {
      seamload_filename = NULL;
      Output(0, "Warning: seam loading not possible with more than 256 images");
    }
    if (is(xor_filename) && n_images > 255) {
      xor_filename.clear();
      Output(
          0, "Warning: XOR map saving not possible with more than 255 images");
    }

    /***********************************************************************
     * Print banner
     ***********************************************************************/
    // Output(1, "\n");
    // Output(1, "Multiblend v2.0.0 (c) 2021 David Horman
    // http://horman.net/mblend/\n"); Output(1,
    // "----------------------------------------------------------------------------\n");

    // ThreadPool *threadpool = ThreadPool::GetInstance(all_threads ? 2 : 0);

    /***********************************************************************
    ************************************************************************
    * Open output
    ************************************************************************
    ***********************************************************************/
    switch (output_type) {
      case ImageType::MB_TIFF: {
        if (!big_tiff)
          tiff_file = TIFFOpen(output_filename, "w");
        else
          tiff_file = TIFFOpen(output_filename, "w8");
        if (!tiff_file)
          die("Error: Could not open output file");
      } break;
      case ImageType::MB_JPEG: {
        if (output_bpp == 16)
          die("Error: 16bpp output is incompatible with JPEG output");
        fopen_s(&jpeg_file, output_filename, "wb");
        if (!jpeg_file)
          die("Error: Could not open output file");
      } break;
      case ImageType::MB_PNG: {
        fopen_s(&jpeg_file, output_filename, "wb");
        if (!jpeg_file)
          die("Error: Could not open output file");
      } break;
      case ImageType::MB_MEM: {
        // fopen_s(&jpeg_file, output_filename, "wb");
        // if (!jpeg_file)
        //   die("Error: Could not open output file");
      } break;
      case ImageType::MB_NONE: {
        assert(false);
      } break;
    }

    int i = 0;

    /***********************************************************************
    ************************************************************************
    * Process images
    ************************************************************************
    ***********************************************************************/
    // timer.Start();

    /***********************************************************************
     * Open images to get prelimary info
     ***********************************************************************/
    size_t untrimmed_bytes = 0;

    for (i = 0; i < n_images; ++i) {
      image_state.images[i]->Open();
      untrimmed_bytes =
          std::max(untrimmed_bytes, image_state.images[i]->untrimmed_bytes);
    }

    /***********************************************************************
     * Check paramters, display warnings
     ***********************************************************************/
    for (i = 1; i < n_images; ++i) {
      if (image_state.images[i]->tiff_xres !=
              image_state.images[0]->tiff_xres ||
          image_state.images[i]->tiff_yres !=
              image_state.images[0]->tiff_yres) {
        Output(
            0,
            "Warning: TIFF resolution mismatch (%f %f/%f %f)\n",
            image_state.images[0]->tiff_xres,
            image_state.images[0]->tiff_yres,
            image_state.images[i]->tiff_xres,
            image_state.images[i]->tiff_yres);
      }
    }

    for (i = 0; i < n_images; ++i) {
      if (output_bpp == 0 && image_state.images[i]->bpp == 16)
        output_bpp = 16;
      if (image_state.images[i]->bpp != image_state.images[0]->bpp) {
        die("Error: mixture of 8bpp and 16bpp image_state.images detected (not currently "
            "handled)\n");
      }
    }

    if (output_bpp == 0)
      output_bpp = 8;
    else if (output_bpp == 16 && output_type == ImageType::MB_JPEG) {
      Output(0, "Warning: 8bpp output forced by JPEG output\n");
      output_bpp = 8;
    }

    /***********************************************************************
     * Allocate working space for reading/trimming/extraction
     ***********************************************************************/
    MapAllocEntryPtr untrimmed_data_entry = MapAlloc::Alloc(untrimmed_bytes);
    void* untrimmed_data = untrimmed_data_entry->data;

    /***********************************************************************
     * Read/trim/extract
     ***********************************************************************/
    for (i = 0; i < n_images; ++i) {
      try {
        image_state.images[i]->Read(untrimmed_data, gamma);
      } catch (char* e) {
        printf("\n\n");
        printf("%s\n", e);
        exit(EXIT_FAILURE);
      }
    }

    /***********************************************************************
     * Clean up
     ***********************************************************************/
    // MapAlloc::Free(untrimmed_data);
    untrimmed_data_entry.reset();

    /***********************************************************************
     * Tighten
     ***********************************************************************/
    // int min_xpos = 0x7fffffff;
    // int min_ypos = 0x7fffffff;
    width = 0;
    height = 0;

    for (i = 0; i < n_images; ++i) {
      min_xpos = std::min(min_xpos, image_state.images[i]->xpos);
      min_ypos = std::min(min_ypos, image_state.images[i]->ypos);
    }

    for (i = 0; i < n_images; ++i) {
      image_state.images[i]->xpos -= min_xpos;
      image_state.images[i]->ypos -= min_ypos;
      width = std::max(
          width, image_state.images[i]->xpos + image_state.images[i]->width);
      height = std::max(
          height, image_state.images[i]->ypos + image_state.images[i]->height);
    }

    // images_time = timer.Read();

    /***********************************************************************
     * Determine number of levels
     ***********************************************************************/
    int blend_wh;
    // int blend_levels;

    if (!fixed_levels) {
      if (!wideblend) {
        std::vector<int> widths;
        std::vector<int> heights;

        for (auto& image : image_state.images) {
          widths.push_back(image->width);
          heights.push_back(image->height);
        }

        std::sort(widths.begin(), widths.end());
        std::sort(heights.begin(), heights.end());

        size_t halfway = (widths.size() - 1) >> 1;

        blend_wh = std::max(
            widths.size() & 1
                ? widths[halfway]
                : (widths[halfway] + widths[halfway + 1] + 1) >> 1,
            heights.size() & 1
                ? heights[halfway]
                : (heights[halfway] + heights[halfway + 1] + 1) >> 1);
      } else {
        blend_wh = (std::max)(width, height);
      }

      blend_levels = (int)floor(log2(blend_wh + 4.0f) - 1);
      if (wideblend)
        blend_levels++;
    } else {
      blend_levels = fixed_levels;
    }

    blend_levels += add_levels;

    if (n_images == 1) {
      blend_levels = 0;
      //   Output(1, "\n%d x %d, %d bpp\n\n", width, height, output_bpp);
    } else {
      //   Output(
      //       1,
      //       "\n%d x %d, %d levels, %d bpp\n\n",
      //       width,
      //       height,
      //       blend_levels,
      //       output_bpp);
    }

    /***********************************************************************
    ************************************************************************
    * Seaming
    ************************************************************************
    ***********************************************************************/
    // timer.Start();

    // Output(1, "Seaming");
    switch (((!seamsave_filename.empty()) << 1) | !!is(xor_filename)) {
      case 1:
        Output(1, " (saving XOR map)");
        break;
      case 2:
        Output(1, " (saving seam map)");
        break;
      case 3:
        Output(1, " (saving XOR and seam maps)");
        break;
    }
    // Output(1, "...\n");

    int min_count;
    int xor_count;
    int xor_image;
    uint64_t utemp;
    int stop;

    uint64_t best;
    uint64_t a, b, c, d;

#define DT_MAX 0x9000000000000000
    uint64_t* prev_line = NULL;
    uint64_t* this_line = NULL;
    bool last_pixel = false;
    bool arbitrary_seam = false;

    std::unique_ptr<Flex> seam_flex = std::make_unique<Flex>(width, height);
    int max_queue = 0;

    int x = 0, y = 0;

    /***********************************************************************
     * Backward distance transform
     ***********************************************************************/
    ThreadPool threadpool(ThreadPool::get_base_thread_pool());
    int n_threads = std::max(2, (int)threadpool.GetNThreads());
    n_threads = std::min((int)kMaxThreadPyramidLineThreads, n_threads);
    std::cout << "Pyramid thread count: " << n_threads << std::endl;
    uint64_t** thread_lines = new uint64_t*[n_threads];

    if (!seamload_filename) {
      std::mutex* flex_mutex_p = new std::mutex;
      std::condition_variable* flex_cond_p = new std::condition_variable;

      uint8_t** thread_comp_lines = new uint8_t*[n_threads];

      for (i = 0; i < n_threads; ++i) {
        thread_lines[i] = new uint64_t[width];
        thread_comp_lines[i] = new uint8_t[width];
      }

      // set all image masks to bottom right
      for (i = 0; i < n_images; ++i) {
        image_state.images[i]->tiff_mask->End();
      }

      for (y = height - 1; y >= 0; --y) {
        int t = y % n_threads;
        this_line = thread_lines[t];
        uint8_t* comp = thread_comp_lines[t];

        // set initial image mask states
        for (i = 0; i < n_images; ++i) {
          image_state.images[i]->mask_state = 0x8000000000000000;
          if (y >= image_state.images[i]->ypos &&
              y < image_state.images[i]->ypos + image_state.images[i]->height) {
            image_state.images[i]->mask_count = width -
                (image_state.images[i]->xpos + image_state.images[i]->width);
            image_state.images[i]->mask_limit = image_state.images[i]->xpos;
          } else {
            image_state.images[i]->mask_count = width;
            image_state.images[i]->mask_limit = width;
          }
        }

        x = width - 1;

        { // make sure the last compression thread to use this chunk of memory
          // is finished
          std::unique_lock<std::mutex> mlock(*flex_mutex_p);
          flex_cond_p->wait(mlock, [&seam_flex, y, n_threads, this] {
            return seam_flex->y > (height - 1) - y - n_threads;
          });
        }

        while (x >= 0) {
          min_count = x + 1;
          xor_count = 0;

          // update image mask states
          for (i = 0; i < n_images; ++i) {
            if (!image_state.images[i]->mask_count) {
              if (x >= image_state.images[i]->mask_limit) {
                utemp = image_state.images[i]->tiff_mask->ReadBackwards32();
                image_state.images[i]->mask_state =
                    ((~utemp) << 32) & 0x8000000000000000;
                image_state.images[i]->mask_count = utemp & 0x7fffffff;
              } else {
                image_state.images[i]->mask_state = 0x8000000000000000;
                image_state.images[i]->mask_count = min_count;
              }
            }

            if (image_state.images[i]->mask_count < min_count)
              min_count = image_state.images[i]->mask_count;
            if (!image_state.images[i]->mask_state) { // mask_state is inverted
              ++xor_count;
              xor_image = i;
            }
          }

          stop = x - min_count;

          if (xor_count == 1) {
            image_state.images[xor_image]->seam_present = true;
            while (x > stop)
              this_line[x--] = xor_image;
          } else {
            if (y == height - 1) { // bottom row
              if (x == width - 1) { // first pixel(s)
                while (x > stop)
                  this_line[x--] = DT_MAX; // max
              } else {
                utemp = this_line[x + 1];
                utemp = MASKVAL(utemp);
                while (x > stop) {
                  utemp += 0x300000000;
                  this_line[x--] = utemp; // was min(temp, DT_MAX) but this is
                                          // unlikely to happen
                }
              }
            } else { // other rows
              if (x == width - 1) { // first pixel(s)
                utemp = prev_line[x - 1] + 0x400000000;
                a = MASKVAL(utemp);

                utemp = prev_line[x] + 0x300000000;
                b = MASKVAL(utemp);

                d = a < b ? a : b;

                this_line[x--] = d;

                if (x == stop) {
                  for (i = 0; i < n_images; ++i) {
                    image_state.images[i]->mask_count -= min_count;
                  }
                  continue;
                }

                c = b + 0x100000000;
                b = a - 0x100000000;
                d += 0x300000000;
              } else {
                utemp = prev_line[x] + 0x300000000;
                b = MASKVAL(utemp);

                utemp = prev_line[x + 1] + 0x400000000;
                c = MASKVAL(utemp);

                utemp = this_line[x + 1] + 0x300000000;
                d = MASKVAL(utemp);
              }

              if (stop == -1) {
                stop = 0;
                last_pixel = true;
              }

              while (x > stop) {
                utemp = prev_line[x - 1] + 0x400000000;
                a = MASKVAL(utemp);

                if (a < d)
                  d = a;
                if (b < d)
                  d = b;
                if (c < d)
                  d = c;

                this_line[x--] = d;

                c = b + 0x100000000;
                b = a - 0x100000000;
                d += 0x300000000;
              }

              if (last_pixel) {
                // d is the new "best" to compare against
                if (b < d)
                  d = b;
                if (c < d)
                  d = c;

                this_line[x--] = d;

                last_pixel = false;
              }
            }
          }

          for (i = 0; i < n_images; ++i) {
            image_state.images[i]->mask_count -= min_count;
          }
        }

        if (y) {
          threadpool.Schedule([this_line,
                               y,
                               comp,
                               h = height,
                               w = width,
                               flex_mutex_p,
                               flex_cond_p,
                               &seam_flex] {
            int p = CompressSeamLine(this_line, comp, w);
            if (p > w) {
              printf("bad p: %d at line %d", p, y);
              exit(0);
            }

            {
              std::unique_lock<std::mutex> mlock(*flex_mutex_p);
              flex_cond_p->wait(mlock, [h, &seam_flex, y] {
                return seam_flex->y == (h - 1) - y;
              });
              seam_flex->Copy(comp, p);
              seam_flex->NextLine();
            }
            flex_cond_p->notify_all();
          });
        }

        prev_line = this_line;
      } // end of row loop

      threadpool.join_all();

      for (i = 0; i < n_images; ++i) {
        if (!image_state.images[i]->seam_present) {
          Output(
              1,
              "Warning: %s is fully obscured by other image_state.images\n",
              image_state.images[i]->filename.c_str());
        }
      }

      for (i = 0; i < n_threads; ++i) {
        if (i >= 2) {
          delete[] thread_lines[i];
          thread_lines[i] = nullptr;
        }
        delete[] thread_comp_lines[i];
      }

      delete[] thread_comp_lines;
      delete flex_mutex_p;
      delete flex_cond_p;
    } else { // if seamload_filename:
      for (i = 0; i < n_images; ++i) {
        image_state.images[i]->tiff_mask->Start();
      }
    }

    // create top level masks
    for (i = 0; i < n_images; ++i) {
      image_state.images[i]->masks.push_back(
          std::make_shared<Flex>(width, height));
    }

    Pnger* xor_map = is(xor_filename)
        ? new Pnger(
              xor_filename.c_str(), "XOR map", width, height, PNG_COLOR_TYPE_PALETTE)
        : NULL;
    Pnger* seam_map = !seamsave_filename.empty()
        ? new Pnger(
              seamsave_filename.c_str(),
              "Seam map",
              width,
              height,
              PNG_COLOR_TYPE_PALETTE)
        : NULL;

    /***********************************************************************
     * Forward distance transform
     ***********************************************************************/
    int current_count = 0;
    std::int64_t current_step;
    std::uint64_t dt_val;

    prev_line = thread_lines[1];

    full_mask_ptr_ = std::make_unique<Flex>(width, height);
    xor_mask_ptr_ = std::make_unique<Flex>(width, height);
    Flex& full_mask = *full_mask_ptr_;
    Flex& xor_mask = *xor_mask_ptr_;

    bool alpha = false;

    for (y = 0; y < height; ++y) {
      for (i = 0; i < n_images; ++i) {
        image_state.images[i]->mask_state = 0x8000000000000000;
        if (y >= image_state.images[i]->ypos &&
            y < image_state.images[i]->ypos + image_state.images[i]->height) {
          image_state.images[i]->mask_count = image_state.images[i]->xpos;
          image_state.images[i]->mask_limit =
              image_state.images[i]->xpos + image_state.images[i]->width;
        } else {
          image_state.images[i]->mask_count = width;
          image_state.images[i]->mask_limit = width;
        }
      }

      x = 0;
      int mc = 0;
      int prev_i = -1;
      int current_i = -1;
      int best_temp;

      while (x < width) {
        min_count = width - x;
        xor_count = 0;

        for (i = 0; i < n_images; ++i) {
          if (!image_state.images[i]->mask_count) {
            if (x < image_state.images[i]->mask_limit) {
              utemp = image_state.images[i]->tiff_mask->ReadForwards32();
              image_state.images[i]->mask_state =
                  ((~utemp) << 32) & 0x8000000000000000;
              image_state.images[i]->mask_count = utemp & 0x7fffffff;
            } else {
              image_state.images[i]->mask_state = 0x8000000000000000;
              image_state.images[i]->mask_count = min_count;
            }
          }

          if (image_state.images[i]->mask_count < min_count)
            min_count = image_state.images[i]->mask_count;
          if (!image_state.images[i]->mask_state) {
            ++xor_count;
            xor_image = i;
          }
        }

        stop = x + min_count;

        if (!xor_count) {
          alpha = true;
        }
        full_mask.MaskWrite(min_count, xor_count);
        xor_mask.MaskWrite(min_count, xor_count == 1);

        if (xor_count == 1) {
          if (xor_map)
            memset(&xor_map->line[x], xor_image, min_count);

          size_t p = (y - image_state.images[xor_image]->ypos) *
                  image_state.images[xor_image]->width +
              (x - image_state.images[xor_image]->xpos);

          int total_count = min_count;
          total_pixels += total_count;
          if (gamma) {
            switch (image_state.images[xor_image]->bpp) {
              case 8: {
                uint16_t v;
                while (total_count--) {
                  v = ((uint8_t*)image_state.images[xor_image]
                           ->channels[0]
                           ->data)[p];
                  channel_totals[0] += v * v;
                  v = ((uint8_t*)image_state.images[xor_image]
                           ->channels[1]
                           ->data)[p];
                  channel_totals[1] += v * v;
                  v = ((uint8_t*)image_state.images[xor_image]
                           ->channels[2]
                           ->data)[p];
                  channel_totals[2] += v * v;
                  ++p;
                }
              } break;
              case 16: {
                uint32_t v;
                while (total_count--) {
                  v = ((uint16_t*)image_state.images[xor_image]
                           ->channels[0]
                           ->data)[p];
                  channel_totals[0] += v * v;
                  v = ((uint16_t*)image_state.images[xor_image]
                           ->channels[1]
                           ->data)[p];
                  channel_totals[1] += v * v;
                  v = ((uint16_t*)image_state.images[xor_image]
                           ->channels[2]
                           ->data)[p];
                  channel_totals[2] += v * v;
                  ++p;
                }
              } break;
            }
          } else {
            switch (image_state.images[xor_image]->bpp) {
              case 8: {
                while (total_count--) {
                  channel_totals[0] += ((uint8_t*)image_state.images[xor_image]
                                            ->channels[0]
                                            ->data)[p];
                  channel_totals[1] += ((uint8_t*)image_state.images[xor_image]
                                            ->channels[1]
                                            ->data)[p];
                  channel_totals[2] += ((uint8_t*)image_state.images[xor_image]
                                            ->channels[2]
                                            ->data)[p];
                  ++p;
                }
              } break;
              case 16: {
                while (total_count--) {
                  channel_totals[0] +=
                      ((uint16_t*)image_state.images[xor_image]
                           ->channels[0]
                           ->data)[p];
                  channel_totals[1] +=
                      ((uint16_t*)image_state.images[xor_image]
                           ->channels[1]
                           ->data)[p];
                  channel_totals[2] +=
                      ((uint16_t*)image_state.images[xor_image]
                           ->channels[2]
                           ->data)[p];
                  ++p;
                }
              } break;
            }
          }

          if (!seamload_filename) {
            RECORD(xor_image, min_count);
            while (x < stop) {
              this_line[x++] = xor_image;
            }
          } else {
            x = stop;
          }

          best = xor_image;
        } else {
          if (xor_map)
            memset(&xor_map->line[x], 0xff, min_count);

          if (!seamload_filename) {
            if (y == 0) {
              // top row
              while (x < stop) {
                best = this_line[x];

                if (x > 0) {
                  utemp = this_line[x - 1] + 0x300000000;
                  d = MASKVAL(utemp);

                  if (d < best)
                    best = d;
                }

                if (best & 0x8000000000000000 && xor_count) {
                  arbitrary_seam = true;
                  for (i = 0; i < n_images; ++i) {
                    if (!image_state.images[i]->mask_state) {
                      best = 0x8000000000000000 | i;
                      if (!reverse)
                        break;
                    }
                  }
                }

                best_temp = best & 0xffffffff;
                RECORD(best_temp, 1);
                this_line[x++] = best;
              }
            } else {
              // other rows
              if (x == 0) {
                SEAM_DT;
                best = dt_val;

                utemp = *prev_line + 0x300000000;
                b = MASKVAL(utemp);
                if (b < best)
                  best = b;

                utemp = prev_line[1] + 0x400000000;
                c = MASKVAL(utemp);
                if (c < best)
                  best = c;

                if (best & 0x8000000000000000 && xor_count) {
                  arbitrary_seam = true;
                  for (i = 0; i < n_images; ++i) {
                    if (!image_state.images[i]->mask_state) {
                      best = 0x8000000000000000 | i;
                      if (!reverse)
                        break;
                    }
                  }
                }

                best_temp = best & 0xffffffff;
                RECORD(best_temp, 1);
                this_line[x++] = best;

                if (x == stop) {
                  for (i = 0; i < n_images; ++i) {
                    image_state.images[i]->mask_count -= min_count;
                  }
                  continue;
                }

                a = b + 0x100000000;
                b = c - 0x100000000;
              } else {
                utemp = prev_line[x - 1] + 0x400000000;
                a = MASKVAL(utemp);

                utemp = prev_line[x] + 0x300000000;
                b = MASKVAL(utemp);
              }

              utemp = best + 0x300000000;
              d = MASKVAL(utemp);

              if (stop == width) {
                stop--;
                last_pixel = true;
              }

              while (x < stop) {
                utemp = prev_line[x + 1] + 0x400000000;
                c = MASKVAL(utemp);

                SEAM_DT;
                best = dt_val;

                if (a < best)
                  best = a;
                if (b < best)
                  best = b;
                if (c < best)
                  best = c;
                if (d < best)
                  best = d;

                if (best & 0x8000000000000000 && xor_count) {
                  arbitrary_seam = true;
                  for (i = 0; i < n_images; ++i) {
                    if (!image_state.images[i]->mask_state) {
                      best = 0x8000000000000000 | i;
                      if (!reverse)
                        break;
                    }
                  }
                }

                best_temp = best & 0xffffffff;
                RECORD(best_temp, 1);
                this_line[x++] = best; // best;

                a = b + 0x100000000;
                b = c - 0x100000000;
                d = best + 0x300000000;
              }

              if (last_pixel) {
                SEAM_DT;
                best = dt_val;

                if (a < best)
                  best = a;
                if (b < best)
                  best = b;
                if (d < best)
                  best = d;

                if (best & 0x8000000000000000 && xor_count) {
                  arbitrary_seam = true;
                  for (i = 0; i < n_images; ++i) {
                    if (!image_state.images[i]->mask_state) {
                      best = 0x8000000000000000 | i;
                      if (!reverse)
                        break;
                    }
                  }
                }

                best_temp = best & 0xffffffff;
                RECORD(best_temp, 1);
                this_line[x++] = best; // best;

                last_pixel = false;
              }
            }
          } else { // if (seamload_filename)...
            x = stop;
          }
        }

        for (i = 0; i < n_images; ++i) {
          image_state.images[i]->mask_count -= min_count;
        }
      }

      if (!seamload_filename) {
        RECORD(-1, 0);

        for (i = 0; i < n_images; ++i) {
          image_state.images[i]->masks[0]->NextLine();
        }
      }

      full_mask.NextLine();
      xor_mask.NextLine();

      if (xor_map)
        xor_map->Write();
      if (seam_map)
        seam_map->Write();

      std::swap(this_line, prev_line);
    }

    if (!seamload_filename) {
      delete[] thread_lines[0];
      delete[] thread_lines[1];
      delete[] thread_lines;
    }

    delete xor_map;
    delete seam_map;

    if (!alpha || output_type == ImageType::MB_JPEG /*||
        output_type == ImageType::MB_MEM*/
        /* arbitrarily have no mask */) {
      no_mask = true;
      // assert(false); // want the mask, yo
    }

    /***********************************************************************
     * Seam load
     ***********************************************************************/
    if (seamload_filename) {
      int png_depth, png_colour;
      png_uint_32 png_width, png_height;
      uint8_t sig[8];
      png_structp png_ptr;
      png_infop info_ptr;
      FILE* f;

      fopen_s(&f, seamload_filename, "rb");
      if (!f)
        die("Error: Couldn't open seam file");

      size_t r =
          fread(sig, 1, 8, f); // assignment suppresses g++ -Ofast warning
      if (!png_check_sig(sig, 8))
        die("Error: Bad PNG signature");

      png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      if (!png_ptr)
        die("Error: Seam PNG problem");
      info_ptr = png_create_info_struct(png_ptr);
      if (!info_ptr)
        die("Error: Seam PNG problem");

      png_init_io(png_ptr, f);
      png_set_sig_bytes(png_ptr, 8);
      png_read_info(png_ptr, info_ptr);
      png_get_IHDR(
          png_ptr,
          info_ptr,
          &png_width,
          &png_height,
          &png_depth,
          &png_colour,
          NULL,
          NULL,
          NULL);

      if (png_width != width || png_height != height)
        die("Error: Seam PNG dimensions don't match workspace");
      if (png_depth != 8 || png_colour != PNG_COLOR_TYPE_PALETTE)
        die("Error: Incorrect seam PNG format");

      png_bytep png_line = (png_bytep)malloc(width);

      for (y = 0; y < height; ++y) {
        png_read_row(png_ptr, png_line, NULL);

        int ms = 0;
        int mc = 0;
        int prev_i = -1;
        int current_i = -1;

        for (x = 0; x < width; ++x) {
          if (png_line[x] > n_images)
            die("Error: Bad pixel found in seam file: %d,%d", x, y);
          RECORD(png_line[x], 1);
        }

        RECORD(-1, 0);

        for (i = 0; i < n_images; ++i) {
          image_state.images[i]->masks[0]->NextLine();
        }
      }

      free(png_line);
    }

    // seam_time = timer.Read();
    //  Do another stage of persistent processing
    more_image_processing(image_state);
    return EXIT_SUCCESS;
  }

  void more_image_processing(const BlenderImageState& image_state) {
    /***********************************************************************
     * Shrink masks
     ***********************************************************************/
    // Output(1, "Shrinking masks...\n");
    // timer.Start();

    ThreadPool threadpool(ThreadPool::get_base_thread_pool());

    const std::size_t n_images = image_state.images.size();
    for (std::size_t i = 0; i < n_images; ++i) {
      threadpool.Schedule([&image_state, i, this] {
        ShrinkMasks(image_state.images[i]->masks, blend_levels);
      });
    }
    threadpool.join_all();

    // shrink_mask_time = timer.Read();
    /***********************************************************************
     * Create shared input pyramids
     ***********************************************************************/
    // wrapping
    wrap_pyramids.clear();
    if (wrap & 1) {
      wrap_levels_h = (int)floor(log2((width >> 1) + 4.0f) - 1);
      wrap_pyramids.push_back(std::make_shared<PyramidWithMasks>(
          width >> 1, height, wrap_levels_h, 0, 0, true));
      wrap_pyramids.push_back(std::make_shared<PyramidWithMasks>(
          (width + 1) >> 1, height, wrap_levels_h, width >> 1, 0, true));
    }

    if (wrap & 2) {
      wrap_levels_v = (int)floor(log2((height >> 1) + 4.0f) - 1);
      wrap_pyramids.push_back(std::make_shared<PyramidWithMasks>(
          width, height >> 1, wrap_levels_v, 0, 0, true));
      wrap_pyramids.push_back(std::make_shared<PyramidWithMasks>(
          width, (height + 1) >> 1, wrap_levels_v, 0, height >> 1, true));
    }

    // masks
    for (auto& py : wrap_pyramids) {
      threadpool.Schedule([=] {
        py->masks.push_back(std::make_shared<Flex>(width, height));
        for (int y = 0; y < height; ++y) {
          if (y < py->GetY() || y >= py->GetY() + py->GetHeight()) {
            py->masks[0]->Write32(0x80000000 | width);
          } else {
            if (py->GetX()) {
              py->masks[0]->Write32(0x80000000 | py->GetX());
              py->masks[0]->Write32(0xc0000000 | py->GetWidth());
            } else {
              py->masks[0]->Write32(0xc0000000 | py->GetWidth());
              if (py->GetWidth() != width)
                py->masks[0]->Write32(0x80000000 | (width - py->GetWidth()));
            }
          }
          py->masks[0]->NextLine();
        }

        ShrinkMasks(
            py->masks, py->GetWidth() == width ? wrap_levels_v : wrap_levels_h);
      });
    }

    threadpool.join_all();
    // end wrapping

    total_levels = std::max({blend_levels, wrap_levels_h, wrap_levels_v, 1});

    // for (int i = 0; i < n_images; ++i) {
    //   image_state.images[i]->pyramid = std::make_shared<Pyramid>(
    //       image_state.images[i]->width,
    //       image_state.images[i]->height,
    //       blend_levels,
    //       image_state.images[i]->xpos,
    //       image_state.images[i]->ypos,
    //       true);
    // }

    // for (int l = total_levels - 1; l >= 0; --l) {
    //   size_t max_bytes = 0;

    //   if (l < blend_levels) {
    //     for (auto& image : image_state.images) {
    //       max_bytes = std::max(max_bytes, image->pyramid->GetLevel(l).bytes);
    //     }
    //   }

    //   for (auto& py : wrap_pyramids) {
    //     if (l < py->GetNLevels())
    //       max_bytes = std::max(max_bytes, py->GetLevel(l).bytes);
    //   }

    //   MapAllocEntryPtr temp_entry;

    //   try {
    //     temp_entry = MapAlloc::Alloc(max_bytes);
    //   } catch (char* e) {
    //     printf("%s\n", e);
    //     exit(EXIT_FAILURE);
    //   }
    //   if (l < blend_levels) {
    //     for (auto& image : image_state.images) {
    //       auto& level = image->pyramid->GetLevel(l);
    //       level.data_item = temp_entry;
    //       level.data = (float*)temp_entry->data;
    //     }
    //   }

    //   for (auto& py : wrap_pyramids) {
    //     if (l < py->GetNLevels()) {
    //       auto& level = py->GetLevel(l);
    //       level.data_item = temp_entry;
    //       level.data = (float*)temp_entry->data;
    //     }
    //   }
    // }
  }

  void setup_image_pyramids(const BlenderImageState& image_state) const {
    std::size_t n_images = image_state.images.size();
    assert(n_images);
    for (int i = 0; i < n_images; ++i) {
      assert(!image_state.images[i]->pyramid);
      image_state.images[i]->pyramid = std::make_shared<Pyramid>(
          image_state.images[i]->width,
          image_state.images[i]->height,
          blend_levels,
          image_state.images[i]->xpos,
          image_state.images[i]->ypos,
          true);
    }

    for (int l = total_levels - 1; l >= 0; --l) {
      size_t max_bytes = 0;

      if (l < blend_levels) {
        for (auto& image : image_state.images) {
          max_bytes = std::max(max_bytes, image->pyramid->GetLevel(l).bytes);
        }
      }

      for (auto& py : wrap_pyramids) {
        if (l < py->GetNLevels())
          max_bytes = std::max(max_bytes, py->GetLevel(l).bytes);
      }

      MapAllocEntryPtr temp_entry;

      try {
        temp_entry = MapAlloc::Alloc(max_bytes);
      } catch (char* e) {
        printf("%s\n", e);
        exit(EXIT_FAILURE);
      }
      if (l < blend_levels) {
        for (auto& image : image_state.images) {
          auto& level = image->pyramid->GetLevel(l);
          level.data_item = temp_entry;
          level.data = (float*)temp_entry->data;
        }
      }

      for (auto& py : wrap_pyramids) {
        if (l < py->GetNLevels()) {
          auto& level = py->GetLevel(l);
          level.data_item = temp_entry;
          level.data = (float*)temp_entry->data;
        }
      }
    }
  }

  int process_inputs(
      const BlenderImageState& image_state,
      std::unique_ptr<hm::MatrixRGB>* output_image) const {
    setup_image_pyramids(image_state);

    ThreadPool threadpool(ThreadPool::get_base_thread_pool());

    //++pass;
    /***********************************************************************
     * No output?
     ***********************************************************************/
    MapAllocEntryPtr output_channel_items[3];
    void* output_channels[3] = {NULL, NULL, NULL};
    int i = 0, x = 0, y = 0;
    int n_images = image_state.images.size();

    /***********************************************************************
     * Create output pyramid
     ***********************************************************************/
    auto output_pyramid =
        std::make_unique<Pyramid>(width, height, total_levels, 0, 0, true);

    for (int l = total_levels - 1; l >= 0; --l) {
      MapAllocEntryPtr temp_entry;
      auto& level = output_pyramid->GetLevel(l);
      try {
        temp_entry = MapAlloc::Alloc(level.bytes);
      } catch (char* e) {
        printf("%s\n", e);
        exit(EXIT_FAILURE);
      }
      level.data_item = temp_entry;
      level.data = (float*)temp_entry->data;
    }

    if (output_type != ImageType::MB_NONE) {
      /***********************************************************************
       * Blend
       ***********************************************************************/
      for (int c = 0; c < 3; ++c) {
        if (n_images > 1) {
          for (i = 0; i < n_images; ++i) {
            // timer.Start();

            image_state.images[i]->pyramid->Copy(
                (uint8_t*)image_state.images[i]->channels[c]->data,
                1,
                image_state.images[i]->width,
                gamma,
                image_state.images[i]->bpp);
            if (output_bpp != image_state.images[i]->bpp)
              image_state.images[i]->pyramid->Multiply(
                  0,
                  gamma ? (output_bpp == 8 ? 1.0f / 66049 : 66049)
                        : (output_bpp == 8 ? 1.0f / 257 : 257));

            image_state.images[i]->channels[c].reset();

            // copy_time += timer.Read();

            // timer.Start();
            image_state.images[i]->pyramid->Shrink();
            // shrink_time += timer.Read();

            // timer.Start();
            image_state.images[i]->pyramid->Laplace();
            // laplace_time += timer.Read();

            // blend into output pyramid...

            // timer.Start();

            for (int l = 0; l < blend_levels; ++l) {
              auto in_level = image_state.images[i]->pyramid->GetLevel(l);
              auto out_level = output_pyramid->GetLevel(l);

              int x_offset = (in_level.x - out_level.x) >> l;
              int y_offset = (in_level.y - out_level.y) >> l;

              for (int b = 0; b < (int)out_level.bands.size() - 1; ++b) {
                int sy = out_level.bands[b];
                int ey = out_level.bands[b + 1];

                threadpool.Schedule([i,
                                     l,
                                     &in_level,
                                     &out_level,
                                     &image_state,
                                     x_offset,
                                     y_offset,
                                     sy,
                                     ey] {
                  for (int y = sy; y < ey; ++y) {
                    int in_line = y - y_offset;
                    if (in_line < 0)
                      in_line = 0;
                    else if (in_line > in_level.height - 1)
                      in_line = in_level.height - 1;
                    float* input_p =
                        in_level.data + (size_t)in_line * in_level.pitch;
                    float* output_p =
                        out_level.data + (size_t)y * out_level.pitch;

                    CompositeLine(
                        input_p,
                        output_p,
                        i,
                        x_offset,
                        in_level.width,
                        out_level.width,
                        out_level.pitch,
                        image_state.images[i]->masks[l]->data,
                        image_state.images[i]->masks[l]->rows[y]);
                  }
                });
              }

              threadpool.join_all();
            }

            // blend_time += timer.Read();
          }

          // timer.Start();
          output_pyramid->Collapse(blend_levels);
          // collapse_time += timer.Read();
        } else {
          assert(false); // why do we care about only one image?
          // timer.Start();

          output_pyramid->Copy(
              (uint8_t*)image_state.images[0]->channels[c]->data,
              1,
              image_state.images[0]->width,
              gamma,
              image_state.images[0]->bpp);
          if (output_bpp != image_state.images[0]->bpp)
            output_pyramid->Multiply(
                0,
                gamma ? (output_bpp == 8 ? 1.0f / 66049 : 66049)
                      : (output_bpp == 8 ? 1.0f / 257 : 257));

          image_state.images[0]->channels[c].reset();

          // copy_time += timer.Read();
        }

        /***********************************************************************
         * Wrapping
         ***********************************************************************/
        if (wrap) {
          assert(false); // never do this at the moment
          // timer.Start();

          int p = 0;

          for (int w = 1; w <= 2; ++w) {
            if (wrap & w) {
              if (w == 1) {
                SwapH(output_pyramid.get());
              } else {
                SwapV(output_pyramid.get());
              }

              int wrap_levels = (w == 1) ? wrap_levels_h : wrap_levels_v;
              for (int wp = 0; wp < 2; ++wp) {
                wrap_pyramids[p]->Copy(
                    (uint8_t *)(output_pyramid->GetData() +
                                wrap_pyramids[p]->GetX() +
                                wrap_pyramids[p]->GetY() *
                                    (std::int64_t)output_pyramid->GetPitch()),
                    1, output_pyramid->GetPitch(), false, 32);
                wrap_pyramids[p]->Shrink();
                wrap_pyramids[p]->Laplace();

                for (int l = 0; l < wrap_levels; ++l) {
                  auto in_level = wrap_pyramids[p]->GetLevel(l);
                  auto out_level = output_pyramid->GetLevel(l);

                  int x_offset = (in_level.x - out_level.x) >> l;
                  int y_offset = (in_level.y - out_level.y) >> l;

                  for (int b = 0; b < (int)out_level.bands.size() - 1; ++b) {
                    int sy = out_level.bands[b];
                    int ey = out_level.bands[b + 1];

                    threadpool.Schedule([=] {
                      for (int y = sy; y < ey; ++y) {
                        int in_line = y - y_offset;
                        if (in_line < 0)
                          in_line = 0;
                        else if (in_line > in_level.height - 1)
                          in_line = in_level.height - 1;
                        float* input_p =
                            in_level.data + (size_t)in_line * in_level.pitch;
                        float* output_p =
                            out_level.data + (size_t)y * out_level.pitch;

                        CompositeLine(
                            input_p,
                            output_p,
                            wp + (l == 0),
                            x_offset,
                            in_level.width,
                            out_level.width,
                            out_level.pitch,
                            wrap_pyramids[p]->masks[l]->data,
                            wrap_pyramids[p]->masks[l]->rows[y]);
                      }
                    });
                  }

                  threadpool.join_all();
                }
                ++p;
              }

              output_pyramid->Collapse(wrap_levels);

              if (w == 1) {
                UnswapH(output_pyramid.get());
              } else {
                UnswapV(output_pyramid.get());
              }
            } // if (wrap & w)
          } // w loop

          // wrap_time += timer.Read();
        }

        /***********************************************************************
         * Offset correction
         ***********************************************************************/
        if (total_pixels) {
          double channel_total = 0; // must be a double
          float* data = output_pyramid->GetData();
          const Flex& xor_mask = *xor_mask_ptr_;
          int mask_pos = 0;
          xor_mask.Start(&mask_pos);

          for (y = 0; y < height; ++y) {
            x = 0;
            while (x < width) {
              uint32_t v = xor_mask.ReadForwards32(&mask_pos);
              if (v & 0x80000000) {
                v = x + v & 0x7fffffff;
                while (x < (int)v) {
                  channel_total += data[x++];
                }
              } else {
                x += v;
              }
            }

            data += output_pyramid->GetPitch();
          }

          float avg = (float)channel_totals[c] / total_pixels;
          if (output_bpp != image_state.images[0]->bpp) {
            switch (output_bpp) {
              case 8:
                avg /= 256;
                break;
              case 16:
                avg *= 256;
                break;
            }
          }
          float output_avg = (float)channel_total / total_pixels;
          output_pyramid->Add(avg - output_avg, 1);
        }

        /***********************************************************************
         * Output
         ***********************************************************************/
        // timer.Start();

        try {
          output_channel_items[c] =
              MapAlloc::Alloc(((size_t)width * height) << (output_bpp >> 4));
          output_channels[c] = output_channel_items[c]->data;
        } catch (char* e) {
          printf("%s\n", e);
          exit(EXIT_FAILURE);
        }

        switch (output_bpp) {
          case 8:
            output_pyramid->Out(
                (uint8_t*)output_channels[c], width, gamma, dither, true);
            break;
          case 16:
            output_pyramid->Out(
                (uint16_t*)output_channels[c], width, gamma, dither, true);
            break;
        }

        // out_time += timer.Read();
      }
      //}
/***********************************************************************
 * Write
 ***********************************************************************/
#define ROWS_PER_STRIP 64
      if (output_filename) {
        Output(1, "Writing %s...\n", output_filename);
      }

      // timer.Start();

      struct jpeg_compress_struct cinfo;
      struct jpeg_error_mgr jerr;

      JSAMPARRAY scanlines = NULL;

      int spp = no_mask ? 3 : 4;

      int bytes_per_pixel = spp << (output_bpp >> 4);
      int bytes_per_row = bytes_per_pixel * width;

      int n_strips = (int)((height + ROWS_PER_STRIP - 1) / ROWS_PER_STRIP);
      int remaining = height;
      auto strip_alloc = std::make_unique<std::uint8_t[]>(
          (ROWS_PER_STRIP * (std::int64_t)width) * bytes_per_pixel);
      // void* strip =
      // malloc((ROWS_PER_STRIP * (std::int64_t)width) * bytes_per_pixel);
      void* strip = strip_alloc.get();
      void* oc_p[3] = {
          output_channels[0], output_channels[1], output_channels[2]};
      if (bgr)
        std::swap(oc_p[0], oc_p[2]);

      std::unique_ptr<Image> output_image_ptr{nullptr};
      std::unique_ptr<Pnger> png_file{nullptr};
      switch (output_type) {
        case ImageType::MB_TIFF: {
          TIFFSetField(tiff_file, TIFFTAG_IMAGEWIDTH, width);
          TIFFSetField(tiff_file, TIFFTAG_IMAGELENGTH, height);
          TIFFSetField(tiff_file, TIFFTAG_COMPRESSION, compression);
          TIFFSetField(tiff_file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
          TIFFSetField(tiff_file, TIFFTAG_ROWSPERSTRIP, ROWS_PER_STRIP);
          TIFFSetField(tiff_file, TIFFTAG_BITSPERSAMPLE, output_bpp);
          if (no_mask) {
            TIFFSetField(tiff_file, TIFFTAG_SAMPLESPERPIXEL, 3);
          } else {
            TIFFSetField(tiff_file, TIFFTAG_SAMPLESPERPIXEL, 4);
            uint16_t out[1] = {EXTRASAMPLE_UNASSALPHA};
            TIFFSetField(tiff_file, TIFFTAG_EXTRASAMPLES, 1, &out);
          }

          TIFFSetField(tiff_file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
          if (image_state.images[0]->tiff_xres != -1) {
            TIFFSetField(
                tiff_file,
                TIFFTAG_XRESOLUTION,
                image_state.images[0]->tiff_xres);
            TIFFSetField(
                tiff_file,
                TIFFTAG_XPOSITION,
                (float)(min_xpos / image_state.images[0]->tiff_xres));
          }
          if (image_state.images[0]->tiff_yres != -1) {
            TIFFSetField(
                tiff_file,
                TIFFTAG_YRESOLUTION,
                image_state.images[0]->tiff_yres);
            TIFFSetField(
                tiff_file,
                TIFFTAG_YPOSITION,
                (float)(min_ypos / image_state.images[0]->tiff_yres));
          }

          if (image_state.images[0]->geotiff.set) {
            // if we got a georeferenced input, store the geotags in the output
            GeoTIFFInfo info(image_state.images[0]->geotiff);
            info.XGeoRef = min_xpos * image_state.images[0]->geotiff.XCellRes;
            info.YGeoRef = -min_ypos * image_state.images[0]->geotiff.YCellRes;
            Output(
                1,
                "Output georef: UL: %f %f, pixel size: %f %f\n",
                info.XGeoRef,
                info.YGeoRef,
                info.XCellRes,
                info.YCellRes);
            geotiff_write(tiff_file, &info);
          }
        } break;
        case ImageType::MB_JPEG: {
          cinfo.err = jpeg_std_error(&jerr);
          jpeg_create_compress(&cinfo);
          jpeg_stdio_dest(&cinfo, jpeg_file);

          cinfo.image_width = width;
          cinfo.image_height = height;
          cinfo.input_components = 3;
          cinfo.in_color_space = JCS_RGB;

          jpeg_set_defaults(&cinfo);
          jpeg_set_quality(&cinfo, jpeg_quality, (boolean) true);
          jpeg_start_compress(&cinfo, (boolean) true);
        } break;
        case ImageType::MB_PNG: {
          assert(false);
          png_file = std::make_unique<Pnger>(
              output_filename,
              nullptr,
              width,
              height,
              no_mask ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGB_ALPHA,
              output_bpp,
              jpeg_file,
              jpeg_quality);
        } break;
        case ImageType::MB_MEM: {
          output_image_ptr = std::make_unique<Image>(
              std::vector<std::size_t>{(std::size_t)width, (std::size_t)height},
              no_mask ? 3 : 4);
        } break;
        case ImageType::MB_NONE: {
          assert(false);
        } break;
      }

      if (output_type == ImageType::MB_PNG ||
          output_type == ImageType::MB_JPEG ||
          output_type == ImageType::MB_MEM) {
        scanlines = new JSAMPROW[ROWS_PER_STRIP];
        for (i = 0; i < ROWS_PER_STRIP; ++i) {
          scanlines[i] = (JSAMPROW) & ((uint8_t*)strip)[i * bytes_per_row];
        }
      }

      const Flex& full_mask = *full_mask_ptr_;
      int mask_pos = 0;
      full_mask.Start(&mask_pos);

      for (int s = 0; s < n_strips; ++s) {
        int strip_p = 0;
        int rows = std::min(remaining, ROWS_PER_STRIP);

        for (int strip_y = 0; strip_y < rows; ++strip_y) {
          x = 0;
          while (x < width) {
            uint32_t cur = full_mask.ReadForwards32(&mask_pos);
            if (cur & 0x80000000) {
              int lim = x + (cur & 0x7fffffff);
              switch (output_bpp) {
                case 8: {
                  while (x < lim) {
                    ((uint8_t*)strip)[strip_p++] = ((uint8_t*)(oc_p[0]))[x];
                    ((uint8_t*)strip)[strip_p++] = ((uint8_t*)(oc_p[1]))[x];
                    ((uint8_t*)strip)[strip_p++] = ((uint8_t*)(oc_p[2]))[x];
                    if (!no_mask)
                      ((uint8_t*)strip)[strip_p++] = 0xff;
                    ++x;
                  }
                } break;
                case 16: {
                  while (x < lim) {
                    ((uint16_t*)strip)[strip_p++] = ((uint16_t*)(oc_p[0]))[x];
                    ((uint16_t*)strip)[strip_p++] = ((uint16_t*)(oc_p[1]))[x];
                    ((uint16_t*)strip)[strip_p++] = ((uint16_t*)(oc_p[2]))[x];
                    if (!no_mask)
                      ((uint16_t*)strip)[strip_p++] = 0xffff;
                    ++x;
                  }
                } break;
              }
            } else {
              size_t t = (size_t)cur * bytes_per_pixel;
              switch (output_bpp) {
                case 8: {
                  ZeroMemory(&((uint8_t*)strip)[strip_p], t);
                } break;
                case 16: {
                  ZeroMemory(&((uint16_t*)strip)[strip_p], t);
                } break;
              }
              strip_p += cur * spp;
              x += cur;
            }
          }

          switch (output_bpp) {
            case 8: {
              oc_p[0] = &((uint8_t*)(oc_p[0]))[width];
              oc_p[1] = &((uint8_t*)(oc_p[1]))[width];
              oc_p[2] = &((uint8_t*)(oc_p[2]))[width];
            } break;
            case 16: {
              oc_p[0] = &((uint16_t*)(oc_p[0]))[width];
              oc_p[1] = &((uint16_t*)(oc_p[1]))[width];
              oc_p[2] = &((uint16_t*)(oc_p[2]))[width];
            } break;
          }
        }
        switch (output_type) {
          case ImageType::MB_TIFF: {
            TIFFWriteEncodedStrip(
                tiff_file, s, strip, rows * (std::int64_t)bytes_per_row);
          } break;
          case ImageType::MB_JPEG: {
            jpeg_write_scanlines(&cinfo, scanlines, rows);
          } break;
          case ImageType::MB_PNG: {
            png_file->WriteRows(scanlines, rows);
          } break;
          case ImageType::MB_MEM: {
            if (!output_image_ptr) {
              die("no target image created");
            }
            output_image_ptr->write_rows(scanlines, rows);
          } break;
          case ImageType::MB_NONE:
          default:
            die("Bad output type)");
            break;
        }

        remaining -= ROWS_PER_STRIP;
      }

      switch (output_type) {
        case ImageType::MB_TIFF: {
          TIFFClose(tiff_file);
        } break;
        case ImageType::MB_JPEG: {
          jpeg_finish_compress(&cinfo);
          jpeg_destroy_compress(&cinfo);
          fclose(jpeg_file);
        } break;
        case ImageType::MB_MEM: {
          if (!output_image_ptr) {
            die("no target image created");
          }
          if (!output_image) {
            die("No output image given");
          }
          auto& img = *output_image_ptr;
          *output_image = std::make_unique<hm::MatrixRGB>(
              img.height,
              img.width,
              img.num_channels(),
              img.consume_raw_data());
        } break;
        case ImageType::MB_PNG:
        case ImageType::MB_NONE: {
          assert(false);
        } break;
      }
      if (scanlines) {
        delete[] scanlines;
        scanlines = nullptr;
      }
      // write_time = timer.Read();
    }

    /***********************************************************************
     * Timing
     ***********************************************************************/
    if (timing) {
      printf("\n");
      printf("Images:   %.3fs\n", images_time);
      printf("Seaming:  %.3fs\n", seam_time);
      if (output_type != ImageType::MB_NONE) {
        printf("Masks:    %.3fs\n", shrink_mask_time);
        printf("Copy:     %.3fs\n", copy_time);
        printf("Shrink:   %.3fs\n", shrink_time);
        printf("Laplace:  %.3fs\n", laplace_time);
        printf("Blend:    %.3fs\n", blend_time);
        printf("Collapse: %.3fs\n", collapse_time);
        if (wrap)
          printf("Wrapping: %.3fs\n", wrap_time);
        printf("Output:   %.3fs\n", out_time);
        printf("Write:    %.3fs\n", write_time);
      }
    }

    /***********************************************************************
     * Clean up
     ***********************************************************************/
    // if (timing) {
    //   if (output_type == ImageType::MB_NONE) {
    //     timer_all.Report("\nExecution complete. Total execution time");
    //   } else {
    //     timer_all.Report("\nBlend complete. Total execution time");
    //   }
    // }

    return EXIT_SUCCESS;
  }
};

// Thread pool replacement
absl::Mutex Blender::thread_pool_mu_;
std::unique_ptr<Eigen::ThreadPool> Blender::thread_pool_;
std::size_t Blender::instance_count_{0};

Blender::Blender() {
  absl::MutexLock lk(&thread_pool_mu_);
  if (!instance_count_++) {
    thread_pool_ = std::make_unique<Eigen::ThreadPool>(
        std::thread::hardware_concurrency() / 2);
  }
}

Blender::~Blender() {
  absl::MutexLock lk(&thread_pool_mu_);
  if (!--instance_count_) {
    thread_pool_.reset();
  }
}

Eigen::ThreadPool* HmThreadPool::get_base_thread_pool() {
  return Blender::gtp();
}

namespace enblend {

int enblend_main(
    std::string output_image_file_name,
    std::vector<std::string> input_files) {
  std::vector<std::string> args;
  args.push_back("python");
  args.push_back("-o");
  args.push_back(output_image_file_name);
  for (const auto& s : input_files) {
    args.push_back(s);
  }

  int argc = args.size();
  char** argv = new char*[argc];

  for (int i = 0; i < argc; ++i) {
    argv[i] = new char[args[i].length() + 1];
    std::strcpy(argv[i], args[i].c_str());
  }
  std::vector<std::reference_wrapper<hm::MatrixRGB>> next_inputs;
  // Call the main function with the converted arguments
  BlenderImageState image_state;
  Blender blender;
  std::unique_ptr<MatrixRGB> output_image;
  int return_value = blender.multiblend_main(argc, argv, image_state);
  return_value = return_value || blender.process_images(image_state);
  return_value = return_value || blender.process_inputs(image_state, nullptr);
  // Clean up the allocated memory
  for (int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;

  return return_value;
}

EnBlender::EnBlender(std::vector<std::string> args) {
  std::vector<std::string> full_args{"python"};
  for (const auto& arg : args) {
    full_args.emplace_back(arg);
  }
  full_args.emplace_back("--all-threads");
  // --no-output must be last
  full_args.emplace_back("--no-output");
  int argc = full_args.size();
  char** argv = new char*[argc];

  for (int i = 0; i < argc; ++i) {
    argv[i] = new char[full_args[i].length() + 1];
    std::strcpy(argv[i], full_args[i].c_str());
  }

  blender_ = std::make_unique<Blender>();
  int result = blender_->multiblend_main(argc, argv, image_state_);
  if (result != EXIT_SUCCESS) {
    std::cerr << "Error initializing blender" << std::endl;
    throw std::runtime_error("Error initializing blender");
  }

  for (int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
}

std::unique_ptr<MatrixRGB> EnBlender::blend_images(
    const std::vector<std::shared_ptr<MatrixRGB>>& images) {
  std::unique_ptr<MatrixRGB> output_image;
  int result = EXIT_SUCCESS;

  std::vector<std::reference_wrapper<hm::MatrixRGB>> blend_images;
  blend_images.reserve(images.size());
  for (auto& img : images) {
    blend_images.emplace_back(*img);
  }
  bool initial_pass = true;
  auto locked = std::make_unique<absl::MutexLock>(&mu_);
  BlenderImageState* current_state = nullptr;
  BlenderImageState new_image_state;
  if (image_state_.images.empty()) {
    image_state_.init_from_images(blend_images);
    current_state = &image_state_;
    // Hold the mutex longer on the first pass
  } else {
    new_image_state.init_from_image_state(image_state_, blend_images);
    current_state = &new_image_state;
    initial_pass = false;
    locked.reset();
  }

  if (initial_pass) {
    result = result || blender_->process_images(*current_state);
    assert(!result);
    locked.reset();
    result = result || blender_->process_inputs(*current_state, &output_image);
  } else {
    result = result || blender_->process_inputs(*current_state, &output_image);
  }
  assert(result == EXIT_SUCCESS);
  return output_image;
}

} // namespace enblend
} // namespace hm
