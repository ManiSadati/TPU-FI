python -m venv pyt-env
source pyt-env/bin/activate

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 17

ln -sf /usr/bin/clang-17 /usr/bin/clang
ln -sf /usr/bin/clang++-17 /usr/bin/clang++

cd /workspace
curl -LO https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-installer-linux-x86_64.sh
chmod +x bazel-6.5.0-installer-linux-x86_64.sh
./bazel-6.5.0-installer-linux-x86_64.sh --user
echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc
echo 'export PATH="$PATH:$HOME/bin"' >> pyt-env/bin/activate
source ~/.bashrc
source pyt-env/bin/activate


apt-get update
apt-get install patchelf
apt update && apt install -y patchelf
pip install Pillow

