FROM nvcr.io/nvidia/pytorch:25.11-py3

ARG TARGETARCH
ARG DEBIAN_FRONTEND=noninteractive

# Core build tooling and runtime libs needed by scikit-sparse / OpenCV / fused-ssim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git wget ca-certificates xz-utils \
    libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev \
    libsuitesparse-dev libmetis-dev liblapack-dev libblas-dev \
    libgl1 libglib2.0-0 colmap \
    && rm -rf /var/lib/apt/lists/*

# Install cuDSS for CUDA 13 (arch-aware)
ARG CUDSS_VERSION=0.7.1.4
ENV CUDSS_ROOT=/opt/cudss-${CUDSS_VERSION}
RUN set -euo pipefail; \
    arch="${TARGETARCH:-$(uname -m)}"; \
    case "${arch}" in \
        amd64|x86_64) cudss_arch="x86_64"; deb_arch_dir="x86_64-linux-gnu";; \
        arm64|aarch64) cudss_arch="aarch64"; deb_arch_dir="aarch64-linux-gnu";; \
        *) echo "Unsupported architecture: ${arch}"; exit 1;; \
    esac; \
    CUDSS_TARBALL="libcudss-linux-${cudss_arch}-${CUDSS_VERSION}_cuda13-archive.tar.xz"; \
    CUDSS_URL_BASE="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-${cudss_arch}"; \
    wget -q "${CUDSS_URL_BASE}/${CUDSS_TARBALL}" -O /tmp/cudss.tar.xz; \
    tar --no-same-owner --no-same-permissions -xJf /tmp/cudss.tar.xz -C /opt; \
    ln -s "/opt/libcudss-linux-${cudss_arch}-${CUDSS_VERSION}_cuda13-archive" "${CUDSS_ROOT}"; \
    mkdir -p /usr/include/libcudss/12 "/usr/lib/${deb_arch_dir}/libcudss/12"; \
    cp -r "${CUDSS_ROOT}/include/"* /usr/include/libcudss/12/; \
    cp -r "${CUDSS_ROOT}/lib/"* "/usr/lib/${deb_arch_dir}/libcudss/12/"; \
    ldconfig; \
    rm /tmp/cudss.tar.xz

# cuDSS toolchain paths
ENV CUDSS_LIBCUDSS_PATHS=/usr/lib/aarch64-linux-gnu/libcudss/12:/usr/lib/x86_64-linux-gnu/libcudss/12
ENV LD_LIBRARY_PATH=${CUDSS_ROOT}/lib:${CUDSS_LIBCUDSS_PATHS}:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDSS_ROOT}/lib:${CUDSS_LIBCUDSS_PATHS}:${LIBRARY_PATH}
ENV CPATH=${CUDSS_ROOT}/include
ENV CUDA_HOME=/usr/local/cuda
ENV FUSED_SSIM_FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0;8.6;8.9;11.0"



CMD ["/bin/bash"]
