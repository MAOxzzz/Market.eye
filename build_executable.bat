@echo off
echo ========================================
echo   Building Market Eye AI Executable
echo ========================================
echo.

:: Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo PyInstaller is not installed. Installing now...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo Failed to install PyInstaller. Please install it manually.
        pause
        exit /b 1
    )
)

:: Make sure we have a clean build
echo Cleaning previous build files...
if exist "dist\Market Eye AI" rmdir /s /q "dist\Market Eye AI"
if exist "build" rmdir /s /q "build"

:: Check for required dependencies
echo Installing required dependencies...
pip install -r requirements.txt
pip install pyinstaller

:: Additional packages that might be missing but required
echo Installing additional packages...
pip install python-multipart email-validator fpdf pyjwt bcrypt passlib

echo Building executable using PyInstaller...
echo This may take several minutes, please be patient...
echo Note: The built executable will show a console window for debugging
echo.

:: Set environment variables for better PyInstaller operation
set PYTHONDONTWRITEBYTECODE=1

:: Create a one-file executable with debug enabled
python -m PyInstaller market_eye.spec --clean

if %errorlevel% neq 0 (
    echo.
    echo Failed to build executable. Check error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Build Complete!
echo ========================================
echo.
echo Executable created in: dist\Market Eye AI\
echo To run the application, go to the dist\Market Eye AI folder
echo and run the "Market Eye AI.exe" file.
echo.
echo IMPORTANT: This version has a console window enabled for debugging.
echo If you encounter high CPU usage or other issues, check the console
echo for error messages.
echo.
echo Default login (after first launch):
echo  - Username: admin
echo  - Password: Password123
echo.
echo Would you like to run the application now? (Y/N)
set /p answer=
if /i "%answer%"=="Y" (
    echo.
    echo Starting Market Eye AI...
    cd dist\Market Eye AI
    start cmd /k "Market Eye AI.exe"
    cd ..\..
)

echo.
pause 