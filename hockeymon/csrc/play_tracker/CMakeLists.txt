set(SOURCES 
  ResizingBox.cpp
  PlayerSTrack.cpp
  PlayTracker.cpp
  PlayTrackerSnapshot.cpp
  PlayDetector.cpp 
  TranslatingBox.cpp
  LivingBox.cpp
  LivingBoxImpl.cpp
  BoxUtils.cpp
)
set(HEADERS
  LivingBox.h
  PlayerSTrack.h
  PlayTracker.h
  PlayTrackerSnapshot.h
  PlayDetector.h
  LivingBoxImpl.h 
  BoxUtils.h
)

add_library(play_tracker STATIC ${SOURCES} ${HEADERS})
