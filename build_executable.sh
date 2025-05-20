#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "  Building Market Eye AI Executable"
echo -e "========================================${NC}"
echo

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller is not installed. Installing now..."
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install PyInstaller. Please install it manually.${NC}"
        exit 1
    fi
fi

# No icon needed - removed icon creation

echo "Building executable using PyInstaller..."
echo "This may take several minutes, please be patient..."
echo

# Create a one-file executable
python3 -m PyInstaller market_eye.spec

if [ $? -ne 0 ]; then
    echo
    echo -e "${RED}Failed to build executable. Check error messages above.${NC}"
    exit 1
fi

echo
echo -e "${GREEN}========================================"
echo "  Build Complete!"
echo -e "========================================${NC}"
echo
echo "Executable created in: dist/Market Eye AI/"
echo "To run the application, go to the dist/Market Eye AI folder"
echo "and run the Market Eye AI executable."
echo

# Ask to run the application
read -p "Would you like to run the application now? (y/n) " answer
if [[ $answer == [Yy]* ]]; then
    echo
    echo "Starting Market Eye AI..."
    cd "dist/Market Eye AI"
    ./Market\ Eye\ AI &
    cd ../..
fi

echo 