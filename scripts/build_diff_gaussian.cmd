@echo on
setlocal EnableExtensions EnableDelayedExpansion

set "CONDA_PREFIX=C:\Users\31791\miniconda3\envs\gs_cuda12"
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%PATH%"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64

set "DISTUTILS_USE_SDK=1"

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"

set "TORCH_CUDA_ARCH_LIST=8.9"

cd /d D:\PKU\thirdyear\mm\Gaussian-Property\external\gaussian-splatting\submodules\diff-gaussian-rasterization

echo CUDA_HOME=%CUDA_HOME%
where cl
where nvcc
where ninja

"%CONDA_PREFIX%\python.exe" setup.py build_ext --inplace -v
