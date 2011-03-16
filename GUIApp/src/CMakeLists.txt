QT4_WRAP_CPP(QT_WRAPS 
  camerawidget.hpp 
  camerasviewwidget.hpp 
  calibrationwidget.hpp
  calibrationdialog.hpp
  errorreport.hpp)

QT4_WRAP_UI(QT_UICS calibrationwidget.ui calibrationdialog.ui)
  
ADD_LIBRARY(GUILib 
  errorreport.hpp
  mainwindow.hpp mainwindow.cpp
  camerawidget.hpp camerawidget.cpp
  camerasviewwidget.hpp camerasviewwidget.cpp
  calibrationwidget.hpp calibrationwidget.cpp
  calibrationdialog.hpp calibrationdialog.cpp
  camcalibrationcontext.hpp camcalibrationcontext.cpp  
  ${QT_WRAPS} ${QT_UICS})

ADD_EXECUTABLE(GUIApp
  main.cpp)

ADD_EXECUTABLE(CalibrationApp
  calibrationapp.cpp)

TARGET_LINK_LIBRARIES(GUIApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
TARGET_LINK_LIBRARIES(CalibrationApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
  
  
  