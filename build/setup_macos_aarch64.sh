#!/bin/bash

# 1. Check Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install Homebrew using the command below..."
    echo "curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
else
    echo "Homebrew already installed."
fi

# 2. Install lightgbm, pytorch, and tensorflow via Homebrew
echo "Installing lightgbm, pytorch, and tensorflow via Homebrew..."
brew install lightgbm pytorch tensorflow

# 3. Download catboost library (.dylib) directly from Github
echo "Downloading CatBoost library..."
wget -q https://github.com/catboost/catboost/releases/download/v1.2.5/libcatboostmodel-darwin-universal2-1.2.5.dylib -O /usr/local/lib/libcatboostmodel.dylib

# 4. Copy lightgbm to /usr/local/lib
echo "Copying LightGBM library to /usr/local/lib..."
# shellcheck disable=SC2046
cp /opt/homebrew/Cellar/lightgbm/$(brew list --versions lightgbm | awk '{print $2}')/lib/lib_lightgbm.dylib /usr/local/lib

# 5. Dynamically get the version of PyTorch installed by Homebrew
PYTORCH_VERSION=$(brew list --versions pytorch | awk '{print $2}')

# 6. Detect the shell and select the correct profile file
if [ -n "$ZSH_VERSION" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
else
    echo "Unsupported shell. Please manually add the environment variables to your shell profile."
    exit 1
fi

# 7. Add environment variables to the profile file
echo "Adding environment variables to $SHELL_PROFILE"

{
    echo "export LIBTORCH=/opt/homebrew/Cellar/pytorch/$PYTORCH_VERSION"
} >> "$SHELL_PROFILE"

echo "Environment variables added to $SHELL_PROFILE. Please run 'source $SHELL_PROFILE' or restart your terminal for the changes to take effect."

echo "Installation and setup complete."