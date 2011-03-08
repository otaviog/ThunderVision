CUDA_ADD_LIBRARY(tdvision  
  pipe.hpp
  workunit.hpp  
  process.hpp
  workunitprocess.hpp workunitprocess.cpp
  cudaprocess.hpp cudaprocess.cpp
  processrunner.hpp processrunner.cpp
  floatimage.hpp floatimage.cpp
  dsimem.hpp dsimem.cpp    
  cuerr.hpp cuerr.cpp
  cudaconstraits.hpp cudaconstraits.cu
  imagereader.hpp imagereader.cpp
  imagewriter.hpp imagewriter.cpp  
  imageresize.hpp imageresize.cpp
  imagesink.hpp imagesink.cpp
  iplimagesink.hpp iplimagesink.cpp  
  rgbconv.hpp rgbconv.cpp
  cpyimagetocpu.hpp cpyimagetocpu.cpp
  medianfiltercpu.hpp medianfiltercpu.cpp
  medianfilterdev.hpp medianfilterdev.cpp medianfilter.cu
  ssddev.hpp ssddev.cpp ssd.cu
  wtadev.hpp wtadev.cpp wta.cu
  benchmark.hpp benchmark.cpp  
  captureproc.hpp captureproc.cpp)

SET(TDVISION_CUDA_SRCS 
  ${CMAKE_CURRENT_SOURCE_DIR}/mediafilter.cu 
  ssd.cu 
  wta.cu)
