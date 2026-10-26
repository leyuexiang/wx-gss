FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    MAX_JOBS=16 \
    TORCH_EXTENSIONS_DIR=/data/.torch_ext \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    QT_QPA_PLATFORM=offscreen

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 构建依赖 + OpenGL/EGL 运行库（不再用 apt 的 CPU 版 colmap）
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git cmake ninja-build build-essential ca-certificates \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev liblz4-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev \
    libgl1 libegl1 libopengl0 libglx0 libglvnd0 libglib2.0-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# PoseLib：COLMAP 默认从 github 拉取（被墙），改用 gitclone.com 镜像预拉，
# 通过 FETCHCONTENT_SOURCE_DIR_POSELIB 让 COLMAP 用本地源码
ARG POSELIB_REPO=https://gitclone.com/github.com/PoseLib/PoseLib.git
ARG POSELIB_COMMIT=8028473d92c9347794a0e3d3541863b5cbb15743
RUN git clone ${POSELIB_REPO} /tmp/poselib && \
    git -C /tmp/poselib checkout ${POSELIB_COMMIT}

# 源码编译 CUDA 版 COLMAP（gitee 镜像）。
# - GPU SIFT 依赖 Qt 的 OpenGLContextManager，必须保持 GUI_ENABLED=ON（默认），
#   运行时配合 QT_QPA_PLATFORM=offscreen + EGL 实现无显示器的 GPU 上下文
# - docker build 阶段 GPU 不可见，CMAKE_CUDA_ARCHITECTURES 不能用 native，显式指定 89（RTX 4060）
ARG COLMAP_REPO=https://gitee.com/mirrors/colmap.git
ARG COLMAP_VERSION=3.11.1
RUN git clone --depth 1 -b ${COLMAP_VERSION} ${COLMAP_REPO} /tmp/colmap && \
    cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DFETCHCONTENT_SOURCE_DIR_POSELIB=/tmp/poselib \
      -DTESTS_ENABLED=OFF && \
    cmake --build /tmp/colmap/build -j16 && \
    cmake --install /tmp/colmap/build && \
    rm -rf /tmp/colmap /tmp/poselib

RUN pip3 config set global.index-url ${PIP_INDEX_URL} && \
    pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 && \
    pip3 install --no-cache-dir gsplat==1.5.3 && \
    pip3 install --no-cache-dir \
        fastapi "uvicorn[standard]" celery redis python-multipart \
        pycolmap scipy pillow numpy pytorch-msssim plyfile

# MASt3R/dust3r 少视角 SfM 后端依赖（源码+权重走 /data 挂载，不打进镜像，见 config.py MAST3R_REPO_DIR）
RUN pip3 install --no-cache-dir \
    einops==0.8.2 roma==1.6.1 opencv-python-headless \
    trimesh==5.0.0 matplotlib==3.10.9 tqdm \
    "huggingface-hub[torch]" safetensors scikit-learn

WORKDIR /app
COPY app/ /app/app/
