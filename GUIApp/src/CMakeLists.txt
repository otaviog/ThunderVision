QT4_WRAP_CPP(QT_WRAPS 
  videowidget.hpp
  #camerasviewwidget.hpp 
  mainwindow.hpp
  camerasviewdialog.hpp
  errorreport.hpp)

QT4_WRAP_UI(QT_UICS 
  calibrationwidget.ui 
  calibrationdialog.ui 
  mainwindow.ui 
  camerasviewdialog.ui)
 
QT4_ADD_RESOURCES(QT_RECS ${CMAKE_CURRENT_SOURCE_DIR}/../resources.qrc)

ADD_LIBRARY(GUILib 
  errorreport.hpp  
  mainwindow.hpp mainwindow.cpp
  videowidget.hpp videowidget.cpp  
  #camerasviewwidget.hpp camerasviewwidget.cpp
  #calibrationwidget.hpp calibrationwidget.cpp
  #calibrationdialog.hpp calibrationdialog.cpp
  #calibrationcontext.hpp calibrationcontext.cpp  
  camerasviewdialog.hpp camerasviewdialog.cpp
  ${QT_WRAPS} ${QT_UICS} ${QT_RECS})

ADD_EXECUTABLE(GUIApp
  main.cpp)

#ADD_EXECUTABLE(CalibrationApp
#  calibrationapp.cpp)

TARGET_LINK_LIBRARIES(GUIApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
#TARGET_LINK_LIBRARIES(CalibrationApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
  
  
  