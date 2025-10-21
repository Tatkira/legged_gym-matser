cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(A1)

#if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

#添加mujoco库（release版）
set(MUJOCO_PATH "/home/one/mujoco210")
include_directories(${MUJOCO_PATH}/include)
link_directories(${MUJOCO_PATH}/bin)
set(MUJOCO_LIB ${MUJOCO_PATH}/bin/libmujoco210.so)

#添加mujoco库（编译安装版）
#set(MUJOCO_FOLDER /opt/mujoco/lib/cmake)
#find_package(mujoco REQUIRED PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)
#if (mujoco_FOUND)
#message(STATUS "Find mujoco: ${mujoco_INCLUDE_DIRS}, ${mujoco_LIBRARIES}")
#endif()

#寻找Eigen3
find_package(Eigen3 REQUIRED)

find_package(Threads REQUIRED)

#添加libtorch库
#set(TORCH_FOLDER /xxx/libtorch-cxx11-abi-shared-with-deps-2.4.1+cu124/libtorch/share/cmake)
set(TORCH_FOLDER /home/one/miniconda3/envs/xdy/lib/python3.8/site-packages/torch/share/cmake)
find_package(Torch REQUIRED PATHS ${TORCH_FOLDER} NO_DEFAULT_PATH)

include_directories(${EIGEN3_INCLUDE_DIR})
add_executable(sim2sim ${PPO_SRC} ${ENV_SRC} ${REPLAY_SRC} sim2sim.cpp)
#编译安装版连接mujoco::mujoco
#target_link_libraries(sim2sim "${TORCH_LIBRARIES}" ${MUJOCO_LIB} glut GL GLU glfw ${EIGEN3_LIBS})

find_library(GLFW_LIB glfw
             NAMES glfw glfw3
             PATHS /usr/lib/x86_64-linux-gnu
             REQUIRED)
             
find_library(GLEW_LIB
             NAMES GLEW glew32 glew
             PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib
             REQUIRED)             

target_link_libraries(sim2sim
    ${TORCH_LIBRARIES}
    ${MUJOCO_LIB}
    glut GL GLU
    /usr/lib/x86_64-linux-gnu/libglfw.so.3
    ${GLFW_LIB}
    ${GLEW_LIB}
    ${EIGEN3_LIBS}
    Threads::Threads
)
