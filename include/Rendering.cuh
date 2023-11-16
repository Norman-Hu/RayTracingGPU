#ifndef RENDERING_H
#define RENDERING_H

#include <Vec3.cuh>
#include <Matrix.cuh>
#include <Scene.cuh>


__global__ void render(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 invViewProj, cudaSurfaceObject_t surface);

// debug
__global__ void testFillFramebuffer(unsigned int w, unsigned int h, cudaSurfaceObject_t surface);
__global__ void renderStraight(Scene * scene, float camNear, Vec3 camPos, Matrix4x4 invViewProj);

#endif // RENDERING_H
