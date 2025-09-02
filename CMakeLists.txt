cmake_minimum_required(VERSION 3.9) # python 3.9 3.10???
project(pinn_loss)

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_library(pinn_loss MODULE pinn_loss.cpp)
target_link_libraries(pinn_loss "${TORCH_LIBRARIES}")
set_property(TARGET pinn_loss PROPERTY CXX_STANDARD 17)

# Ensure the produced module is importable as `import pinn_loss` from Python:
# - remove the automatic `lib` prefix
# - force `.so` suffix (on macOS this will still be a loadable module)
# - place the output in the project source directory so `import` can find it
set_target_properties(pinn_loss PROPERTIES PREFIX "" OUTPUT_NAME "pinn_loss" SUFFIX ".so")
set_target_properties(pinn_loss PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")


