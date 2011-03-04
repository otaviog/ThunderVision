CUDA_ADD_LIBRARY(tdvision
  floatimage.hpp floatimage.cpp
  imagefilterwu.hpp
  medianfilterwucpu.hpp medianfilterwucpu.cpp
  medianfilterwudev.hpp medianfilterwudev.cpp medianfilter.cu
  cudaconstraits.hpp cudaconstraits.cu
  cuerr.hpp cuerr.cpp
  workunit.hpp  
  workunitrunner.hpp workunitrunner.cpp
  pipe.hpp
  capturewu.hpp capturewu.cpp
  ssdwudev.hpp ssdwudev.cpp ssd.cu
  dsimem.hpp dsimem.cpp
  resizeimagewu.hpp resizeimagewu.cpp
  imagereaderwu.hpp imagereaderwu.cpp
  imagewriterwu.hpp imagewriterwu.cpp
  imagesink.hpp imagesink.cpp
  rgbconv.hpp rgbconv.cpp
  iplimagesink.hpp iplimagesink.cpp
  benchmark.hpp benchmark.cpp
  )
