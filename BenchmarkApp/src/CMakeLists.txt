ADD_LIBRARY(BenchmarkLib 
  ibenchmarkdataset.hpp
  benchmarkrunner.hpp benchmarkrunner.cpp
  benchdatasetfactory.hpp benchdatasetfactory.cpp
  imatchercompmetric.hpp
  meancompmetric.hpp meancompmetric.cpp)

ADD_EXECUTABLE(BenchmarkApp main.cpp)

TARGET_LINK_LIBRARIES(BenchmarkApp BenchmarkLib ${ThunderVision_LIBRARIES})