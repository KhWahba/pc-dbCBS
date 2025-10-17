cmake_minimum_required(VERSION 3.5)

project(pc-dbCBS)

find_package(Boost 1.58 REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YAML REQUIRED yaml-cpp)
find_package(fcl REQUIRED)

find_package(NLopt REQUIRED)
message(STATUS "Found NLopt in ${NLOPT_INCLUDE_DIRS}")

add_subdirectory(deps/dynoplan)

set(MJ_ROOT "$ENV{HOME}/inria/phdv2025-khaled_wahba/src/deps/mujoco-3.3.6")
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