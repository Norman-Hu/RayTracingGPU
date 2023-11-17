#ifndef MATERIAL_H
#define MATERIAL_H


#include <Vec3.cuh>


struct BlinnPhongMaterial
{
	Vec3 ambient;
	Vec3 diffuse;
	Vec3 specular;
	float shininess;

	float mirror;
	float glass;
};

#endif // MATERIAL_H
