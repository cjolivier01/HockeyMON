set(SOURCES 
  kmeans.cpp
  file_io.cpp
  wtime.cpp
  seq_kmeans.cpp
  omp_kmeans.cpp
  wtime.cpp
)

set(HEADERS kmeans.h)

add_library(kmeans STATIC ${SOURCES} ${HEADERS})
