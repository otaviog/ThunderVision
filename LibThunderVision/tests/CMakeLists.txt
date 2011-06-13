INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src)

ADD_LIBRARY(StereoTest stereotest.hpp stereotest.cpp)

MACRO(ADD_TEST output source)
  CUDA_ADD_EXECUTABLE(${output} ${source} tdvtestmain.cpp)
  TARGET_LINK_LIBRARIES(${output}
    ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} StereoTest ${ThunderVision_LIBRARIES})
ENDMACRO(ADD_TEST)

ADD_TEST(testmedianfilter testmedianfilter.cpp)
ADD_TEST(testpipeline testpipeline.cpp)
ADD_TEST(testssddev testssddev.cpp)
ADD_TEST(testrectification testrectification.cpp)
ADD_TEST(testthunderlang testthunderlang.cpp)
ADD_TEST(testtdvcontext testtdvcontext.cpp)
ADD_TEST(testcapture testcapture.cpp)
ADD_TEST(testcalibration testcalibration.cpp)
ADD_TEST(testcrosscorrelation testcrosscorrelation.cpp)
ADD_TEST(testbirchfieldcost testbirchfieldcost.cpp)
ADD_TEST(testsemiglobal testsemiglobal.cpp)
ADD_TEST(testsemiglobalcpu testsemiglobalcpu.cpp)


CUDA_ADD_EXECUTABLE(testgridglmesh testgridglmesh.cpp)
TARGET_LINK_LIBRARIES(testgridglmesh
  ${ThunderVision_LIBRARIES} ${GLUT_LIBRARIES})

