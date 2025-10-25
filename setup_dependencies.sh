#!/bin/bash
# 依赖环境快速修复脚本

echo "=== 创建extern目录结构 ==="
mkdir -p /home/one/isaac/legged_gym-master/extern
cd /home/one/isaac/legged_gym-master/extern

echo "=== 下载OSQP依赖 ==="
if [ ! -d "osqp" ]; then
    git clone https://github.com/osqp/osqp.git
    cd osqp
    git checkout v0.6.2
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=/home/one/isaac/legged_gym-master/extern/osqp/install ..
    make -j4
    make install
    cd /home/one/isaac/legged_gym-master/extern
fi

if [ ! -d "eigen3" ]; then
    echo "=== 下载Eigen3依赖 ==="
    git clone https://gitlab.com/libeigen/eigen.git
    cd eigen
    git checkout 3.4.0
    cd /home/one/isaac/legged_gym-master/extern
fi

echo "=== 依赖安装完成 ==="