QT4_WRAP_CPP(QT_WRAPS 
  videowidget.hpp
  mainwindow.hpp
  camerasviewdialog.hpp
  calibrationdialog.hpp
  selectinputdialog.hpp
  errorreport.hpp
  rectificationviewdialog.hpp
  disparitydialog.hpp
  reprojectionview.hpp
  thanksdialog.hpp
  )

QT4_WRAP_UI(QT_UICS 
  calibrationdialog.ui 
  mainwindow.ui 
  camerasviewdialog.ui
  selectinputdialog.ui
  rectificationviewdialog.ui
  disparitydialog.ui
  thanks.ui
  )
 
QT4_ADD_RESOURCES(QT_RECS ${CMAKE_CURRENT_SOURCE_DIR}/../resources.qrc)

ADD_LIBRARY(GUILib 
  errorreport.hpp  
  cmdline.hpp cmdline.cpp
  mainwindow.hpp mainwindow.cpp
  videowidget.hpp videowidget.cpp  
  calibrationdialog.hpp calibrationdialog.cpp
  camerasviewdialog.hpp camerasviewdialog.cpp
  selectinputdialog.hpp selectinputdialog.cpp
  rectificationviewdialog.hpp rectificationviewdialog.cpp
  disparitydialog.hpp disparitydialog.cpp
  reprojectionview.hpp reprojectionview.cpp
  thanksdialog.hpp thanksdialog.cpp
  ${QT_WRAPS} ${QT_UICS} ${QT_RECS})

ADD_EXECUTABLE(GUIApp
  main.cpp)

ADD_EXECUTABLE(CalibrationApp
  calibrationapp.cpp)

TARGET_LINK_LIBRARIES(GUIApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
TARGET_LINK_LIBRARIES(CalibrationApp GUILib ${QT_LIBRARIES} ${ThunderVision_LIBRARIES})
  
  
  