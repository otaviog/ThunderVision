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
  capturewu.hpp capturewu.cpp)
