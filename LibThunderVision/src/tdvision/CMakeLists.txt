ADD_ANTLR3_GRAMMAR(ThunderLang)

CUDA_ADD_LIBRARY(tdvision  
  dim.hpp dim.cpp
  pipe.hpp
  workunit.hpp  
  process.hpp
  misc.hpp misc.cpp
  tmpbufferimage.hpp tmpbufferimage.cpp
  workunitprocess.hpp workunitprocess.cpp
  cudaprocess.hpp cudaprocess.cpp
  processgroup.hpp processgroup.cpp
  exceptionreport.hpp
  updatecount.hpp updatecount.cpp
  processrunner.hpp processrunner.cpp
  floatimage.hpp floatimage.cpp
  dsimemutil.h
  dsimem.hpp dsimem.cpp    
  cuerr.hpp cuerr.cpp
  cudaconstraits.hpp cudaconstraits.cu
  sink.hpp sink.cpp
  imagereader.hpp imagereader.cpp
  imagewriter.hpp imagewriter.cpp  
  imageresize.hpp imageresize.cpp
  rgbconv.hpp rgbconv.cpp
  floatconv.hpp floatconv.cpp
  cpyimagetocpu.hpp cpyimagetocpu.cpp
  teeworkunit.hpp 
  medianfiltercpu.hpp medianfiltercpu.cpp
  medianfilterdev.hpp medianfilterdev.cpp medianfilter.cu
  matchingcost.hpp matchingcost.cpp
  ssddev.hpp ssddev.cpp ssd.cu  
  crosscorrelationdev.hpp crosscorrelationdev.cpp crosscorrelation.cu
  mutualinformationdev.hpp mutualinformationdev.cpp mutualinformation.cu
  optimizer.hpp optimizer.cpp
  wtadev.hpp wtadev.cpp wta.cu
  dynamicprogdev.hpp dynamicprogdev.cpp dynamicprog.cu
  dynamicprogcpu.hpp dynamicprogcpu.cpp
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
  stereomatcherfactory.hpp
  commonstereomatcherfactory.hpp commonstereomatcherfactory.cpp
  thunderlang.hpp thunderlang.cpp
  tlctx.h tlctx.cpp  
  ThunderLangParser.h ThunderLangParser.c
  ThunderLangLexer.h ThunderLangLexer.c
  parserexception.hpp
  glcommon.hpp
  vertexbuffer.hpp vertexbuffer.cpp
  reprojector.hpp 
  reprojection.hpp
  cvreprojector.hpp cvreprojector.cpp
  directreprojector.hpp
  gridglmesh.hpp gridglmesh.cpp
  glreprojection.hpp glreprojection.cpp
  reprojectprocess.hpp reprojectprocess.cpp
  )
