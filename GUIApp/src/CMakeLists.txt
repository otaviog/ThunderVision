QT4_WRAP_CPP(QT_WRAPS camerawidget.hpp camerasviewwidget.hpp 
  calibrationwidget.hpp)

ADD_LIBRARY(GUILib 
  mainwindow.hpp mainwindow.cpp
  camerawidget.hpp camerawidget.cpp
  camerasviewwidget.hpp camerasviewwidget.cpp
  calibrationwidget.hpp calibrationwidget.cpp
  ${QT_WRAPS})

ADD_EXECUTABLE(GUIApp
  main.cpp)

ADD_EXECUTABLE(CalibrationApp
  calibrationapp.cpp)

TARGET_LINK_LIBRARIES(GUIApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
TARGET_LINK_LIBRARIES(CalibrationApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
  
  
  