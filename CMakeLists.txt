cmake_minimum_required(VERSION 3.5)

project(pc-dbCBS)

find_package(Boost 1.58 REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YAML REQUIRED yaml-cpp)
find_package(fcl REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(NLopt REQUIRED)
message(STATUS "Found NLopt in ${NLOPT_INCLUDE_DIRS}")

# Build position-independent code so static libs can be linked into shared libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(deps/dynoplan)

# --- MuJoCo root (portable) ---
set(MJ_ROOT "" CACHE PATH "Path to MuJoCo root (contains include/ and lib/)")

if(NOT MJ_ROOT)
  if(DEFINED ENV{MJ_ROOT})
    set(MJ_ROOT "$ENV{MJ_ROOT}")
  else()
    set(MJ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/mujoco-3.3.6")
  endif()
endif()

message(STATUS "MJ_ROOT = ${MJ_ROOT}")

if(NOT EXISTS "${MJ_ROOT}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "MuJoCo headers not found: ${MJ_ROOT}/include/mujoco/mujoco.h")
endif()

if(NOT EXISTS "${MJ_ROOT}/lib/libmujoco.so")
  message(FATAL_ERROR "MuJoCo library not found: ${MJ_ROOT}/lib/libmujoco.so")
endif()

if(NOT glfw3_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLFW REQUIRED glfw3) # uses glfw3.pc
endif()


# Enable C++17 and warnings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")



include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${FCL_INCLUDE_DIRS}
)



add_executable(pc_dbcbs
  src/pc_dbcbs.cpp
  src/init_guess_mujoco.cpp
  src/init_guess_payload.cpp
  src/init_guess_unicycles.cpp
)

target_include_directories(pc_dbcbs PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/dynoplan/include
)


target_link_libraries(pc_dbcbs PRIVATE 
  idbastar::optimization
  idbastar::tdbastar
  ${Boost_LIBRARIES}
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
  ${NLOPT_LIBRARIES} 
  glfw
  )


add_library(pcdbcbs_api SHARED
  src/pc_dbcbs_api.cpp
  src/init_guess_mujoco.cpp
  src/init_guess_payload.cpp
  src/init_guess_unicycles.cpp
)

target_include_directories(pcdbcbs_api PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/dynoplan/include
)

target_link_libraries(pcdbcbs_api PRIVATE
  idbastar::optimization
  idbastar::tdbastar
  ${Boost_LIBRARIES}
  ${YAML_LIBRARIES}
  ${FCL_LIBRARIES}
  ${NLOPT_LIBRARIES}
  glfw
)

# Python module (pybind11 is already added by dynobench)
pybind11_add_module(pcdbcbs MODULE
  src/py_pcdbcbs.cpp
)

target_include_directories(pcdbcbs PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/dynoplan/include
)

target_link_libraries(pcdbcbs PRIVATE
  pcdbcbs_api
)
