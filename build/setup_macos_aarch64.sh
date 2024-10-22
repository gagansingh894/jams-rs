#!/bin/bash

# 1. Install Homebrew if not already installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
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

# 6. Add environment variables to ~/.bash_profile or ~/.zshrc based on the shell
echo "Setting environment variables..."

SHELL_PROFILE="~/.bash_profile"
if [ "$SHELL" = "/bin/zsh" ]; then
    SHELL_PROFILE="~/.zshrc"
fi

{
    echo "export LIBTORCH=/opt/homebrew/Cellar/pytorch/$PYTORCH_VERSION"
    echo "export LIGHTGBM_LIB_PATH=/opt/homebrew/Cellar/lightgbm/$(brew list --versions lightgbm | awk '{print $2}')/lib/"
    echo "export DYLD_LIBRARY_PATH=/usr/local/lib:\$DYLD_LIBRARY_PATH"
} >> $SHELL_PROFILE

echo "Environment variables added to $SHELL_PROFILE. Please run 'source $SHELL_PROFILE' or restart your terminal for the changes to take effect."

echo "Installation and setup complete."