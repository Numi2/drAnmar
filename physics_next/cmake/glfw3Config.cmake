# DrAnmar bridge for CRESSim's upstream `find_package(glfw3)` tests.
# GLFW_LIBRARY must point to a runtime-provided libglfw shared object.
if(NOT GLFW_LIBRARY OR NOT EXISTS "${GLFW_LIBRARY}")
    message(FATAL_ERROR "GLFW_LIBRARY must name an existing GLFW shared library")
endif()
if(NOT TARGET glfw)
    add_library(glfw SHARED IMPORTED)
    set_target_properties(glfw PROPERTIES IMPORTED_LOCATION "${GLFW_LIBRARY}")
endif()
