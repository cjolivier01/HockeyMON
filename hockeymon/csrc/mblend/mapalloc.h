#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#ifdef _WIN32
#include <Windows.h>
#endif
namespace hm {

#define _MAPALLOC_

class MapAlloc {
  MapAlloc();
  ~MapAlloc();
  class MapAllocObject {
  public:
    MapAllocObject(std::size_t _size, int alignment);
    ~MapAllocObject();
    void* GetPointer();
    std::size_t GetSize() { return size; }
    bool IsFile();

  private:
#ifdef _WIN32
    HANDLE file = NULL;
    HANDLE map = NULL;
#else
    int file = 0;
#endif
    void* pointer{nullptr};
    std::size_t size{0};
  };
  //static std::vector<MapAllocObject*> objects;
  static std::unordered_map<const void *, std::unique_ptr<MapAllocObject>> object_map;
  static char tmpdir[8192];
  //static char filename[8192];
  static int suffix;
  static std::size_t cache_threshold;
  static std::size_t total_allocated;

public:

  struct MapAllocEntry {
    void *data{nullptr};
    ~MapAllocEntry() {
      if (data) {
        MapAlloc::Free(data);
      }
    }
  };

  static std::shared_ptr<MapAllocEntry> Alloc(std::size_t size, int alignment = 16);
  static std::size_t GetSize(const void *p);
  static void CacheThreshold(std::size_t threshold);
  static void SetTmpdir(const char* _tmpdir);
  //static bool LastFile();
//  static bool last_mapped;
private:
  static void Free(void* p);
};

using MapAllocEntry = MapAlloc::MapAllocEntry;
using MapAllocEntryPtr = std::shared_ptr<MapAllocEntry>;

}
