#!/bin/bash
set -e
APP_SRC=$(find ~/Library/Developer/Xcode/DerivedData/Hex-*/Build/Products/Release/Hex.app -maxdepth 0 2>/dev/null | head -1)
if [ -z "$APP_SRC" ]; then
    echo "No Release build found. Build first with:"
    echo "  cd ~/projects/caspi-hebrew-asr/Hex && xcodebuild build -scheme Hex -configuration Release -skipMacroValidation CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual"
    exit 1
fi
echo "Installing from $APP_SRC"
rsync -a --delete "$APP_SRC/" /Applications/Hex.app/
echo "Installed. Permissions preserved."
