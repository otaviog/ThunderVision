ADD_LIBRARY(tinyxml 
  tinystr.cpp
  tinystr.h
  tinyxml.cpp
  tinyxmlerror.cpp
  tinyxml.h
  tinyxmlparser.cpp)

FILE(GLOB incs ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
INSTALL(FILES ${incs} DESTINATION include/ud)
INSTALL(TARGETS tinyxml
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)