ADD_LIBRARY(BenchmarkLib 
  bmdataset.hpp
  bmrunner.hpp bmrunner.cpp
  bmdatasetfactory.hpp bmdatasetfactory.cpp
  qualitymetric.hpp
  meanqmetric.hpp meanqmetric.cpp)

ADD_EXECUTABLE(BenchmarkApp main.cpp)
TARGET_LINK_LIBRARIES(BenchmarkApp BenchmarkLib ${ThunderVision_LIBRARIES})

#ADD_EXECUTABLE(SimpleBM simplebenchmark.cpp)
#TARGET_LINK_LIBRARIES(SimpleBM BenchmarkLib ${ThunderVision_LIBRARIES})