ADD_LIBRARY(GUILib 
  mainwindow.hpp mainwindow.cpp
  camerawidget.hpp camerawidget.cpp)

ADD_EXECUTABLE(GUIApp
  main.cpp)

TARGET_LINK_LIBRARIES(GUIApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
  
  
  