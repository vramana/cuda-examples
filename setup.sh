git clone https://github.com/vramana/cuda-examples.git

cd cuda-examples

touch Visualization.ipynb

mkdir -p build

uv sync

cd build

cmake ..


# git pull && cmake --build . && ./main
