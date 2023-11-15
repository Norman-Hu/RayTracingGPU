#include <Scene.cuh>

__device__ Scene::Scene()
: objectList(nullptr)
, size(0)
{

}

__device__ Scene::Scene(Hitable ** _objectList, int _size)
: objectList(_objectList)
, size(_size)
{
}

__device__ Scene::~Scene()
{
	for (int i=0; i<size; ++i)
		delete objectList[i];
	delete [] objectList;
}

__device__ bool Scene::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
	bool hasHit = false;
	float closest_t = tmax;

	Hit tmpHit;
	for (int i=0; i<size; ++i)
	{
		if (objectList[i]->hit(ray, tmin, tmax, tmpHit))
		{
			hasHit = true;
			if (tmpHit.t < closest_t)
			{
				closest_t = tmpHit.t;
				out = tmpHit;
			}
		}
	}
	return hasHit;
}
