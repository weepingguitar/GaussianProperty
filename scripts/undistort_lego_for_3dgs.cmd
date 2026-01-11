@echo off
setlocal

REM Undistort the lego COLMAP model into a 3DGS-compatible layout.
REM Output will be created at: gp_cases_dirs\lego_3dgs

set REPO_ROOT=%~dp0\..
for %%I in ("%REPO_ROOT%") do set REPO_ROOT=%%~fI

set IN_SCENE=%REPO_ROOT%\gp_cases_dirs\lego
set OUT_SCENE=%REPO_ROOT%\gp_cases_dirs\lego_3dgs

echo Input:  %IN_SCENE%
echo Output: %OUT_SCENE%

python %REPO_ROOT%\reconstruction\undistort_for_3dgs.py --scene_root "%IN_SCENE%" --out_root "%OUT_SCENE%" --force

endlocal
