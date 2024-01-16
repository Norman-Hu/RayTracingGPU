IMPORTANT : initialiser les submodules git avant de compiler
git submodule update --init --recursive
puis
mkdir build && cd build
cmake ..
make raytracer2

# Raytracer CUDA

Controls:
- X: toggle depth rendering
- WASD: move
- Q/E: go up/down
- Esc: free mouse cursor

Usage:
./raytracer2 <scene.glb>
