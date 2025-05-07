cmake_minimum_required(VERSION 3.20...3.27)
project(agony)

# Set build type to Release
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

# nanobind_add_module(${PROJECT_NAME} src/agony.cc)
nanobind_add_module(${PROJECT_NAME} src/agony.cpp)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${BINDIR}
)