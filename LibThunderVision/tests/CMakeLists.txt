INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src)

MACRO(ADD_TEST output source)
ADD_EXECUTABLE(${output} ${source})
TARGET_LINK_LIBRARIES(${output}
  ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${ThunderVision_LIBRARIES})
ENDMACRO(ADD_TEST)

ADD_TEST(testmedianfilter testmedianfilter.cpp)
ADD_TEST(testpipeline testpipeline.cpp)