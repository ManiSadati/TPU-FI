cd /home/TPU-FI/third_party/tensorflow
bazel build -j $(nproc) //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow_cpu
pip uninstall tensorflow-cpu -y
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp310-cp310-linux_x86_64.whl
pip install pandas

