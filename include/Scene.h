#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <Object.h>

class Scene
{
public:
	Scene();
	~Scene();

private:
	std::vector<Object> contents;
};

#endif // SCENE_H
