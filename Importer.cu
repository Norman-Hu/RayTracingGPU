#include <Importer.cuh>
#include <Scene.cuh>
#include <Object.cuh>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <CudaHelpers.h>
#include <BVH.cuh>
#include <vector>
#include <deque>


Scene * importSceneToGPU(const std::string & file)
{
	Assimp::Importer importer;

	const aiScene * scene = importer.ReadFile(file,
											  aiProcess_CalcTangentSpace |
											  aiProcess_Triangulate |
											  aiProcess_JoinIdenticalVertices |
											  aiProcess_SortByPType |
//											  aiProcess_PreTransformVertices |
											  aiProcess_FindInstances);

	if (nullptr == scene)
	{
		std::cerr << "Error reading file for scene " << file << "." << std::endl;
		return nullptr;
	}

	if (scene->mNumMeshes == 0)
	{
		std::cerr << "Error reading scene " << file << " : no meshes." << std::endl;
		return nullptr;
	}

	Scene * d_scene = createScene();
	syncAndCheckErrors();


	BVH * bvhList = new BVH[scene->mNumMeshes];
	Mesh * meshes = new Mesh[scene->mNumMeshes];
	for (int idMesh=0; idMesh<scene->mNumMeshes; ++idMesh)
	{
		const aiMesh & aimesh = *scene->mMeshes[idMesh];
		Mesh mesh;
		mesh.vertices = new Vec3[aimesh.mNumVertices*sizeof(Vec3)];
		mesh.normals = new Vec3[aimesh.mNumVertices*sizeof(Vec3)];
		memcpy(mesh.vertices, aimesh.mVertices, aimesh.mNumVertices*sizeof(Vec3));
		memcpy(mesh.normals, aimesh.mNormals, aimesh.mNumVertices*sizeof(Vec3));
		mesh.indices = new unsigned int[aimesh.mNumFaces*3];
		for (int i=0; i<aimesh.mNumFaces; ++i)
		{
			mesh.indices[3*i+0] = aimesh.mFaces[i].mIndices[0];
			mesh.indices[3*i+1] = aimesh.mFaces[i].mIndices[1];
			mesh.indices[3*i+2] = aimesh.mFaces[i].mIndices[2];
		}

		meshes[idMesh] = std::move(mesh);

		// build a BVH
		BVH bvh(idMesh, meshes[idMesh]);
		bvhList[idMesh] = std::move(bvh);
	}

	// Copy BVH list to GPU
	{
		BVH * d_bvhList = Scene::createBVHList(d_scene, scene->mNumMeshes);
		for (int i=0; i<scene->mNumMeshes; ++i)
			BVH::copyToGPU(bvhList[i], d_bvhList+i);
	}


    // Create BVH instances
    std::vector<BVHInstance> instances;
    {
        std::deque<aiNode*> queue;
        queue.push_back(scene->mRootNode);
        while (!queue.empty())
        {
            aiNode * n = queue.front();
            queue.pop_front();
            for (int c=0; c<n->mNumChildren; ++c)
                queue.push_back(n->mChildren[c]);

            if (n->mNumMeshes == 0)
                continue;

            BVHInstance instance;
            instance.bvhID = n->mMeshes[0];

            aiMatrix4x4 transfo = n->mTransformation;
            aiMatrix4x4 inverseTransfo = n->mTransformation;
            inverseTransfo.Inverse();
            transfo.Transpose();
            inverseTransfo.Transpose();
            std::memcpy(instance.transform.data(), &transfo.a1, sizeof(float)*16);
            std::memcpy(instance.invTransform.data(), &inverseTransfo.a1, sizeof(float)*16);

            Vec3 min = bvhList[instance.bvhID].nodes[0].aabb.min;
            Vec3 max = bvhList[instance.bvhID].nodes[0].aabb.max;
            for (int i = 0; i < 8; i++)
            {
                Vec4 transformed = instance.transform * Vec4(i & 1 ? max.x : min.x, i & 2 ? max.y : min.y, i & 4 ? max.z : min.z, 1.0f);
                instance.bounds.grow_vec3(Vec3{transformed.x, transformed.y, transformed.z});
            }
            instances.push_back(std::move(instance));
        }
    }

    // Copy BVH instances to GPU
    {
        BVHInstance * d_bvhInstanceList = Scene::createBVHInstanceList(d_scene, instances.size());
        for (int i=0; i<instances.size(); ++i)
            BVHInstance::copyToGPU(instances[i], d_bvhInstanceList+i);
    }


	// TODO: Setup the TLAS
	{
		TLAS tlas();

	}

	setLightCount(d_scene, scene->mNumLights);
	syncAndCheckErrors();
	for (int idLight=0; idLight<scene->mNumLights; ++idLight)
	{
		const aiLight & light = *scene->mLights[idLight];
		Light * d_light = createPointLight({light.mPosition.x, light.mPosition.y, light.mPosition.z});
		syncAndCheckErrors();
		setLight(d_scene, idLight, d_light);
		syncAndCheckErrors();
	}

	setMaterialCount(d_scene, scene->mNumMaterials);
	syncAndCheckErrors();
	for (int idMat=0; idMat<scene->mNumMaterials; ++idMat)
	{
		const aiMaterial & mat = *scene->mMaterials[idMat];
		PBRMaterial material;
		aiColor3D base;
		if (mat.Get(AI_MATKEY_BASE_COLOR, base) != AI_SUCCESS)
			std::cerr << "Error: no diffuse" << std::endl;
		material.albedo = {base.r, base.g, base.b};
		float rough;
		if (mat.Get(AI_MATKEY_ROUGHNESS_FACTOR, rough) != AI_SUCCESS)
			std::cerr << "Error: no roughness" << std::endl;
		material.roughness = rough;
		float metallic;
		if (mat.Get(AI_MATKEY_METALLIC_FACTOR, metallic) != AI_SUCCESS)
			std::cerr << "Error: no metallic" << std::endl;
		material.metallic = rough;
		setMaterial(d_scene, idMat, material);
		syncAndCheckErrors();
	}


	return d_scene;
}
