#ifndef OBJECT_H
#define OBJECT_H

#include <vector>
#include <Vec3.h>

enum class ObjectType
{
	Sphere,
	Plane,
	Mesh
};

class Object
{
public:
	ObjectType type;
	union
	{
		struct {
			std::vector<Vec3> vertices;
			std::vector<unsigned int> indices;
		} mesh;
		struct {
			Vec3 c;
			float r;
		} sphere;
		struct {
			Vec3 p, n;
		} plane;
	};
};

#endif // OBJECT_H
