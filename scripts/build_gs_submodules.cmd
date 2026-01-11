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
set "PIP_NO_BUILD_ISOLATION=1"

cd /d D:\PKU\thirdyear\mm\Gaussian-Property\external\gaussian-splatting

where cl
where nvcc
where ninja
"%CONDA_PREFIX%\python.exe" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())"

"%CONDA_PREFIX%\python.exe" -m pip install -v submodules\diff-gaussian-rasterization --no-build-isolation
"%CONDA_PREFIX%\python.exe" -m pip install -v submodules\simple-knn --no-build-isolation
"%CONDA_PREFIX%\python.exe" -m pip install -v submodules\fused-ssim --no-build-isolation

"%CONDA_PREFIX%\python.exe" -c "import diff_gaussian_rasterization, simple_knn, fused_ssim; print('imports_ok')"
