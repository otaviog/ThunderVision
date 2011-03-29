ADD_ANTLR3_GRAMMAR(ThunderLang)

CUDA_ADD_LIBRARY(tdvision  
  dim.hpp dim.cpp
  pipe.hpp
  workunit.hpp  
  process.hpp
  workunitprocess.hpp workunitprocess.cpp
  cudaprocess.hpp cudaprocess.cpp
  processgroup.hpp processgroup.cpp
  exceptionreport.hpp
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
  floatconv.hpp floatconv.cpp
  cpyimagetocpu.hpp cpyimagetocpu.cpp
  teeworkunit.hpp 
  medianfiltercpu.hpp medianfiltercpu.cpp
  medianfilterdev.hpp medianfilterdev.cpp medianfilter.cu
  matchingcost.hpp
  ssddev.hpp ssddev.cpp ssd.cu
  optimizer.hpp
  wtadev.hpp wtadev.cpp wta.cu
  benchmark.hpp benchmark.cpp  
  captureproc.hpp captureproc.cpp
  stereoinputsource.hpp
  capturestereoinputsource.hpp capturestereoinputsource.cpp
  camerasdesc.hpp camerasdesc.cpp
  cameraparameters.hpp cameraparameters.cpp
  chessboardpattern.hpp chessboardpattern.cpp
  rectificationcv.hpp rectificationcv.cpp
  calibration.hpp calibration.cpp
  stereocorrespondencecv.hpp stereocorrespondencecv.cpp
  tdvcontext.hpp tdvcontext.cpp
  reconstruction.hpp reconstruction.cpp
  ctrlprocess.hpp ctrlprocess.cpp
  stereomatcher.hpp
  devstereomatcher.hpp devstereomatcher.cpp
  cpustereomatcher.hpp cpustereomatcher.cpp
  thunderlang.hpp thunderlang.cpp
  tlctx.h tlctx.cpp  
  ThunderLangParser.h ThunderLangParser.c
  ThunderLangLexer.h ThunderLangLexer.c
  parserexception.hpp)