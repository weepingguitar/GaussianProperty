@echo off
setlocal

REM Train 3D Gaussian Splatting on the undistorted lego dataset.
REM Requires: conda env "gs_cuda12" with PyTorch CUDA installed + 3DGS submodules built.

set REPO_ROOT=%~dp0\..
for %%I in ("%REPO_ROOT%") do set REPO_ROOT=%%~fI

set SCENE=%REPO_ROOT%\gp_cases_dirs\lego_3dgs
set OUT=%REPO_ROOT%\Results_3dgs_lego_7000_cmd

echo Scene: %SCENE%
echo Out:   %OUT%

call conda activate gs_cuda12
if errorlevel 1 (
  echo Failed to activate conda env gs_cuda12
  exit /b 1
)

cd /d %REPO_ROOT%\external\gaussian-splatting

python train.py -s "%SCENE%" -m "%OUT%" --iterations 7000 --save_iterations 7000 --test_iterations 7000 --disable_viewer

endlocal
