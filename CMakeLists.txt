cmake_minimum_required(VERSION 3.10)
project(pinn_loss)

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

add_library(pinn_loss MODULE pinn_loss.cpp)
target_link_libraries(pinn_loss PRIVATE ${TORCH_LIBRARIES} pybind11::module)
set_property(TARGET pinn_loss PROPERTY CXX_STANDARD 17)

# Ensure the produced module is importable as `import pinn_loss` from Python
set_target_properties(pinn_loss PROPERTIES PREFIX "" OUTPUT_NAME "pinn_loss" SUFFIX ".so")
set_target_properties(pinn_loss PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Optional: Debugging output
message(STATUS "Torch version: ${TORCH_VERSION}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
message(STATUS "pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Output directory: ${CMAKE_SOURCE_DIR}")

