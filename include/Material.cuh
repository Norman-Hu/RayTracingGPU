#ifndef MATERIAL_H
#define MATERIAL_H


#include <Vec3.cuh>


struct BlinnPhongMaterial
{
	Vec3 ambient;
	Vec3 diffuse;
	Vec3 specular;
	float shininess;

	bool refraction;

	float mirror;
	float refractiveIndex;
};

#endif // MATERIAL_H
