# Orign: groundstation project
# Find the FTGL Library

FIND_PATH(FTGL_INCLUDE_DIR FTFont.h
  /usr/local/include
  /usr/include
  /usr/local/FTGL
  )

# The FTGL include and build paths aren't always nicely related
# so toss the library option up as well

FIND_LIBRARY(FTGL_LIBRARY NAMES ftgl_static_MT ftgl
  PATHS /usr/lib /usr/local/lib)

# Let other code know we found FTGL

IF(FTGL_INCLUDE_DIR)
  IF(FTGL_LIBRARY)
 	# The fact that it is in the cache is deprecated.
 	SET (FTGL_FOUND 1 CACHE INTERNAL "FTGL library and headers are available")
  ENDIF(FTGL_LIBRARY)
ENDIF(FTGL_INCLUDE_DIR)
