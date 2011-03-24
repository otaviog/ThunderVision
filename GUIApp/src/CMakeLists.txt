QT4_WRAP_CPP(QT_WRAPS 
  videowidget.hpp
  #camerasviewwidget.hpp 
  calibrationwidget.hpp
  calibrationdialog.hpp
  mainwindow.hpp
  errorreport.hpp)

QT4_WRAP_UI(QT_UICS calibrationwidget.ui calibrationdialog.ui mainwindow.ui)
  
ADD_LIBRARY(GUILib 
  errorreport.hpp  
  inputprocess.hpp
  camerasinputprocess.hpp camerasinputprocess.cpp
  mainwindow.hpp mainwindow.cpp
  videowidget.hpp videowidget.cpp  
  #camerasviewwidget.hpp camerasviewwidget.cpp
  calibrationwidget.hpp calibrationwidget.cpp
  calibrationdialog.hpp calibrationdialog.cpp
  calibrationcontext.hpp calibrationcontext.cpp  
  ${QT_WRAPS} ${QT_UICS})

ADD_EXECUTABLE(GUIApp
  main.cpp)

ADD_EXECUTABLE(CalibrationApp
  calibrationapp.cpp)

TARGET_LINK_LIBRARIES(GUIApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
TARGET_LINK_LIBRARIES(CalibrationApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
  
  
  