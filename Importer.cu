#include <Importer.cuh>
#include <Scene.cuh>
#include <Object.cuh>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <CudaHelpers.h>



Scene * importSceneToGPU(const std::string & file)
{
	Assimp::Importer importer;

	const aiScene * scene = importer.ReadFile(file,
											  aiProcess_CalcTangentSpace |
											  aiProcess_Triangulate |
											  aiProcess_JoinIdenticalVertices |
											  aiProcess_SortByPType |
											  aiProcess_PreTransformVertices);

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
	setHitableCount(d_scene, scene->mNumMeshes);
	syncAndCheckErrors();

	for (int idMesh=0; idMesh<scene->mNumMeshes; ++idMesh)
	{
		const aiMesh & mesh = *scene->mMeshes[idMesh];
		Mesh * d_mesh = createMesh();
		setMeshAABB(d_mesh, computeAABB((Vec3*)&(mesh.mVertices[0].x), mesh.mNumVertices));
		setMeshVertices(d_mesh, (Vec3*)&(mesh.mVertices[0].x), mesh.mNumVertices);
		setMeshMaterial(d_mesh, mesh.mMaterialIndex);
		syncAndCheckErrors();
		setMeshNormals(d_mesh, (Vec3*)&(mesh.mNormals[0].x), mesh.mNumVertices);
		syncAndCheckErrors();
		unsigned int * indices = new unsigned int[mesh.mNumFaces*3];
		for (int i=0; i<mesh.mNumFaces; ++i)
		{
			indices[3*i+0] = mesh.mFaces[i].mIndices[0];
			indices[3*i+1] = mesh.mFaces[i].mIndices[1];
			indices[3*i+2] = mesh.mFaces[i].mIndices[2];
		}
		setMeshIndices(d_mesh, indices, mesh.mNumFaces*3);
		syncAndCheckErrors();
		delete [] indices;

		setHitable(d_scene, idMesh, d_mesh);
		syncAndCheckErrors();
	}

	setLightCount(d_scene, scene->mNumLights);
	syncAndCheckErrors();
	for (int idLight=0; idLight<scene->mNumLights; ++idLight)
	{
		const aiLight & light = *scene->mLights[idLight];
        Vec3 color = {light.mColorDiffuse[0], light.mColorDiffuse[1], light.mColorDiffuse[2]};
        color /= color.max();
		Light * d_light = createPointLight({light.mPosition.x, light.mPosition.y, light.mPosition.z}, color);
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
		material.metallic = metallic;
        aiColor3D emissive(0.f, 0.f, 0.f);
        if (mat.Get(AI_MATKEY_COLOR_EMISSIVE, emissive) != AI_SUCCESS)
            std::cerr << "Error: no emissive" << std::endl;
        material.emissive = {emissive.r, emissive.g, emissive.b};
        float transmissive = 0.0f;
        mat.Get(AI_MATKEY_TRANSMISSION_FACTOR, transmissive);
        material.transmissive = transmissive;

		setMaterial(d_scene, idMat, material);
		syncAndCheckErrors();
	}


	return d_scene;
}
