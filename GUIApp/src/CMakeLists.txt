QT4_WRAP_CPP(QT_WRAPS 
  videowidget.hpp
  mainwindow.hpp
  camerasviewdialog.hpp
  calibrationwidget.hpp 
  calibrationdialog.hpp
  selectinputdialog.hpp
  errorreport.hpp)

QT4_WRAP_UI(QT_UICS 
  calibrationwidget.ui 
  calibrationdialog.ui 
  mainwindow.ui 
  camerasviewdialog.ui
  selectinputdialog.ui)
 
QT4_ADD_RESOURCES(QT_RECS ${CMAKE_CURRENT_SOURCE_DIR}/../resources.qrc)

ADD_LIBRARY(GUILib 
  errorreport.hpp  
  cmdline.hpp cmdline.cpp
  mainwindow.hpp mainwindow.cpp
  videowidget.hpp videowidget.cpp  
  calibrationwidget.hpp calibrationwidget.cpp
  calibrationdialog.hpp calibrationdialog.cpp
  camerasviewdialog.hpp camerasviewdialog.cpp
  selectinputdialog.hpp selectinputdialog.cpp
  ${QT_WRAPS} ${QT_UICS} ${QT_RECS})

ADD_EXECUTABLE(GUIApp
  main.cpp)

#ADD_EXECUTABLE(CalibrationApp
#  calibrationapp.cpp)

TARGET_LINK_LIBRARIES(GUIApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
#TARGET_LINK_LIBRARIES(CalibrationApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
  
  
  