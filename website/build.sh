#!/bin/bash

# Exit on errors
set -e

# Create dist folder if it doesn't exist
mkdir -p dist

# Download the ZIP file
ZIP_URL="https://github.com/zeroby0/AdaptDCT/archive/refs/heads/main.zip"
ZIP_FILE="dist/AdaptDCT.zip"

echo "Downloading ZIP file..."
curl -L -o "$ZIP_FILE" "$ZIP_URL"

cp website.html dist/index.html

echo "Build complete!"
