INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src)

MACRO(ADD_TEST output source)
  CUDA_ADD_EXECUTABLE(${output} ${source})
  TARGET_LINK_LIBRARIES(${output}
    ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${ThunderVision_LIBRARIES})
ENDMACRO(ADD_TEST)

ADD_TEST(testmedianfilter testmedianfilter.cpp)
ADD_TEST(testpipeline testpipeline.cpp)
ADD_TEST(testssddev testssddev.cpp)
ADD_TEST(testrectification testrectification.cpp)
ADD_TEST(testthunderlang testthunderlang.cpp)
ADD_TEST(testtdvcontext testtdvcontext.cpp)
ADD_TEST(testcapture testcapture.cpp)
ADD_TEST(testcalibration testcalibration.cpp)