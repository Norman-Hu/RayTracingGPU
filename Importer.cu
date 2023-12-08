#include <Importer.cuh>
#include <Scene.cuh>
#include <Object.cuh>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <CudaHelpers.h>
#include <BVH.cuh>



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

	std::vector<BVHInstance>

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
