cmake_minimum_required(VERSION 3.19)
project(vsdpir CXX)

build_command(target clean)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH "${TORCH_LIB_PATH};${BOOST_LIB_PATH}")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

string(REGEX REPLACE "([\\/\\-]O)3" "\\12" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "([\\/\\-]O)3" "\\12" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REGEX REPLACE "([\\/\\-]O)3" "\\12" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")
string(REGEX REPLACE "([\\/\\-]O)3" "\\12" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

add_library(
  ${PROJECT_NAME} SHARED
  src/vsdpir.cpp
  src/commons.cpp
  src/pFrame.cpp
  src/UNetRes.cpp
)

find_library(vapoursynth "${VS_LIB_PATH}\\sdk\\lib64")

target_include_directories(${PROJECT_NAME} PRIVATE "${BOOST_LIB_PATH}")
target_include_directories(${PROJECT_NAME} PRIVATE "${VS_LIB_PATH}\\sdk\\include")

find_package(Torch REQUIRED)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${TORCH_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
)