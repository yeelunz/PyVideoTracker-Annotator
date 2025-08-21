@echo off
echo 正在使用 PyInstaller 打包 PyVideoTracker-Annotator...
echo.

:: 檢查是否已安裝依賴
echo 檢查依賴套件...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo 正在安裝 PyInstaller...
    pip install pyinstaller
)

pip show opencv-contrib-python >nul 2>&1
if errorlevel 1 (
    echo 正在安裝 opencv-contrib-python...
    pip install opencv-contrib-python
)

pip show PySide6 >nul 2>&1
if errorlevel 1 (
    echo 正在安裝 PySide6...
    pip install PySide6
)

pip show numpy >nul 2>&1
if errorlevel 1 (
    echo 正在安裝 numpy...
    pip install numpy
)

:: 清理之前的打包結果
echo 清理之前的打包結果...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

:: 使用 spec 文件進行打包
echo 開始打包...
pyinstaller --clean PyVideoTracker-Annotator.spec

if exist "dist\PyVideoTracker-Annotator\PyVideoTracker-Annotator.exe" (
    echo.
    echo 打包成功！
    echo 執行檔位置: dist\PyVideoTracker-Annotator\PyVideoTracker-Annotator.exe
    echo.
    echo 您可以將整個 dist\PyVideoTracker-Annotator 資料夾複製到其他電腦使用
) else (
    echo.
    echo 打包失敗，請檢查錯誤訊息
)

pause
