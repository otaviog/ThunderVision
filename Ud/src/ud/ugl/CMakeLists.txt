INCLUDE_DIRECTORIES(${UD_UGL_INCLUDE_DIR})
ADD_LIBRARY(UdUgl
  aabb.cpp aabb.hpp
  collision.hpp collision.cpp
  fpscounter.hpp fpscounter.cpp
  framebuffer.hpp framebuffer.cpp
  misc.hpp misc.cpp
  light.hpp light.cpp
  color.hpp color.cpp
  pathanimation.hpp pathanimation.cpp
  frustum.hpp frustum.cpp
  boundingsphere.hpp
  projection.cpp projection.hpp
  camera.hpp camera.cpp
  glstate.hpp
  stereo.hpp stereo.cpp)

TARGET_LINK_LIBRARIES(UdUgl ${UD_COMMON_LIBRARIES} ${UD_MATH_LIBRARIES} ${UD_UGL_LIBRARIES} UdCommon UdMath)

FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
INSTALL(FILES ${files} DESTINATION include/ud/ugl)

INSTALL(TARGETS UdUgl
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
