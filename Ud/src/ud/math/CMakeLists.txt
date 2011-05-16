INCLUDE_DIRECTORIES(${UD_MATH_INCLUDE_DIR})
ADD_LIBRARY(UdMath
  math.hpp math.cpp
  matrix22.hpp
  matrix33.hpp
  matrix44.hpp
  quat.hpp quat.cpp
  plane.hpp
  vec.hpp
  vec2.hpp #vec2d.hpp
  vec3.hpp
  vec4.hpp)

FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
INSTALL(FILES ${files} DESTINATION include/ud/math)
INSTALL(TARGETS UdMath
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
