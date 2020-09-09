## jeston nano上运行安装

- 安装tensorflow

参考：https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

`pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==1.15.2+nv20.4`

- 安装opencv

参考：https://www.pyimagesearch.com/2020/03/25/how-to-configure-your-nvidia-jetson-nano-for-computer-vision-and-deep-learning/

`sudo apt-get install libavcodec-dev  libavformat-dev libswscale-dev  libxvidcore-dev libavresample-dev libjpeg-dev libpng-dev libcanberra-gtk-module libcanberra-gtk3-module libtiff-dev  python-tk libgtk-3-dev libv4l-dev libdc1394-22-dev
apt install libhdf5-dev libv4l-dev v4l-utils qv4l2 v4l2ucp libdc1394-22-dev`

参考：https://pysource.com/2019/08/26/install-opencv-4-1-on-nvidia-jetson-nano/

```
cmake     -D WITH_CUDA=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules \
        -D WITH_GSTREAMER=ON \
        -D WITH_LIBV4L=ON \
        -D BUILD_opencv_python2=ON \
        -D BUILD_opencv_python3=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local ..
```

## 运行依赖
- `cuda`
- `tensorlfow-1.15`
- `opencv`
- `GPU`
- `Ubuntu18.04`

## 推理

`python3 demo.py --style_model models/tangyan.pb `
