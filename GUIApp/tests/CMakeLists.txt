INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../src)

MACRO(ADD_QTEST output source)
ADD_EXECUTABLE(${output} ${source})
TARGET_LINK_LIBRARIES(${output}
  GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
ENDMACRO(ADD_QTEST)

ADD_QTEST(testcamerawidget testcamerawidget.cpp)