#include <Rendering.cuh>
#include <surface_indirect_functions.h>
#include <cstdio>

#define PI 3.14159265358979323846f
#define INV_PI 0.31830988618379067154f
#define INV_2PI 0.15915494309189533577f

// brdf
__host__ __device__ float distributionGGX(const Vec3 & N, const Vec3 & H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = fmaxf(Vec3::dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

__host__ __device__ float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r*r) / 8.0f;

    float num   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return num / denom;
}

__host__ __device__ float geometrySmith(const Vec3 & N, const Vec3 & V, const Vec3 & L, float roughness)
{
    float NdotV = fmaxf(Vec3::dot(N, V), 0.0f);
    float NdotL = fmaxf(Vec3::dot(N, L), 0.0f);
    float ggx2  = geometrySchlickGGX(NdotV, roughness);
    float ggx1  = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

__host__ __device__ Vec3 fresnelSchlick(float cosTheta, const Vec3 & F0)
{
    float tmp = 1.0f - cosTheta;
    tmp = tmp < 0.0f ? 0.0f : tmp;
    tmp = tmp > 1.0f ? 1.0f : tmp;
    return F0 + Vec3{1.0f-F0.x, 1.0f-F0.y, 1.0f-F0.z} * powf(tmp, 5.0f);
}

__host__ __device__ Vec3 sampleUniformHemisphere(float u, float v)
{
    float z = u;
    float r = sqrtf(1.0f - z*z);
    float phi = 2.0f * PI * v;
    return {r * cosf(phi), r * sinf(phi), z};
}

__host__ __device__ Vec3 sampleUniformSphere(float u, float v)
{
    float z = 1.0f - 2.0f * u;
    float r = sqrtf(1.0f - z*z);
    float phi = 2.0f * PI * v;
    return {r * cosf(phi), r * sinf(phi), z};
}


__global__ void render(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface, curandState_t * randState, bool sampleLights)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = gridDim.x*blockDim.x*(blockDim.y*blockIdx.y+threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	if (x<w && y<h)
	{
        // Find ray
		float ndc_x = (2.f*(float)x / (float)w)-1.f;
		float ndc_y = 1.f - (2.f*(float)y / (float)h);
		Vec4 vec{ndc_x, ndc_y, camNear, 1.f};
		vec = rayTransform*vec;
		Vec3 dir = Vec3{vec[0], vec[1], vec[2]};
		dir.normalize();
		Ray ray{camPos, dir.normalized()};

        // init
        Vec3 L{0.0f, 0.0f, 0.0f};
        Vec3 beta{1.0f, 1.0f, 1.0f};
		constexpr int maxBounces = 5;
        bool specularBounce = true;
        constexpr int maxDepth = 5;
        int depth = 0;

		while (beta.max() > 0.0f)
        {
            Hit hitInfo;
            bool hit = scene->hit(ray, 0.001f, 100.0f, hitInfo);
            if (!hit) {
                // infinite lights, which have a fixed color
                if (!sampleLights || specularBounce)
                    L += compwise_mul(beta, {.2f, .2f, .2f});
                break;
            }

            // Get material
            PBRMaterial &mat = scene->materials[hitInfo.materialId];

            // if we don't sample lights directly, account for emissive surfaces
            if (!sampleLights || specularBounce)
            {
                L += compwise_mul(beta, mat.emissive);
            }

            // end path if max depth is reached
            if (depth++ == maxDepth)
                break;

            // direct illumination
            Vec3 wo = -ray.direction;
            if (sampleLights)
            {

            }

            if (false) // TODO: Importance sampling
            {

            }
            else
            {
                // Uniform sampling
                Vec3 wi;
                float pdf = INV_2PI;
                if (false) // TODO: reflective+transmissive
                {

                }
                else
                {
                    wi = sampleUniformHemisphere(curand_uniform(randState+tid), curand_uniform(randState+tid));
                    pdf = INV_2PI;
                    // reflection
                    if (Vec3::dot(wo, hitInfo.normal) * Vec3::dot(wi, hitInfo.normal) < 0)
                        wi = -wi;
                    // transmission
//                    else if (Vec3::dot(wo, hitInfo.normal) * Vec3::dot(wi, hitInfo.normal) > 0)
//                        wi = -wi;
                }

                Vec3 H = (wo+wi); H.normalize();
                Vec3 F0 = Vec3{0.04f, 0.04f, 0.04f};
                F0      = Vec3::mix(F0, mat.albedo, mat.metallic);
                Vec3 F  = fresnelSchlick(fmaxf(Vec3::dot(H, wo), 0.0), F0);
                float NDF = distributionGGX(hitInfo.normal, H, mat.roughness);
                float G   = geometrySmith(hitInfo.normal, wo, wi, mat.roughness);
                Vec3 numerator    = NDF * G * F;
                float denominator = 4.0f * fmaxf(Vec3::dot(hitInfo.normal, wo), 0.0f) * fmaxf(Vec3::dot(hitInfo.normal, wi), 0.0f)  + 0.0001f;
                Vec3 specular     = numerator / denominator;
                Vec3 kS = F;
                Vec3 kD = Vec3{1.0f, 1.0f, 1.0f} - kS;
                kD *= 1.0f - mat.metallic;

                beta = compwise_mul(specular * fabsf(Vec3::dot(wi, hitInfo.normal)) / pdf, beta);
                specularBounce = false;
                ray.origin = hitInfo.p;
                ray.direction = wi;
            }

//			if (hit)
//			{
//				int lightHits = 0;
//				for (int l=0; l<lightsCount; l++)
//				{
//					Light * light = scene->lights[l];
//					LightSamples samples = light->getSamples(randState);
//
//					int sampleHits = 0;
//
//					Vec3 colorForLight(0.0f, 0.0f, 0.0f);
//					for (int lightSample=0; lightSample<samples.size; ++lightSample)
//					{
//						Vec3 lightPos = samples.samples[lightSample];
//
//						Vec3 lightDir = (lightPos - hitInfo.p);
//						float lightDistance = lightDir.length();
//						lightDir.normalize();
//
//						// check if obstructed
//						Hit _unused;
//						if (scene->hit({hitInfo.p, lightDir}, 0.001f, lightDistance, _unused))
//							continue;
//
//						++sampleHits;
//
////						float diff = max(Vec3::dot(hitInfo.normal, lightDir), 0.0f);
////						Vec3 diffuse = mat.diffuse * diff;
////
////						Vec3 viewDir = (-ray.direction).normalized();
////						Vec3 halfDir = (lightDir + viewDir).normalized();
////						float spec = powf(max(Vec3::dot(hitInfo.normal, halfDir), 0.0f), mat.shininess);
////						Vec3 specular = spec * mat.specular;
////
////						Vec3 res = mat.ambient + diffuse + specular;
////						colorForLight += res.mulComp(light->color);
//						colorForLight += mat.albedo;
//					}
//					if (sampleHits > 0)
//					{
//						colorForLight/=sampleHits;
//						++lightHits;
//						color += colorForLight;
//					}
//				}
//				if (lightHits > 0)
//					color /= lightHits;
//				break;
//			}
//			else break;
//		}
//
//		float maxVal = max(color.x, max(color.y, color.z));
//		if (maxVal > 1.0f)
//			color /= maxVal;
//		color *= 255.0f;
//		val.x = color.x;
//		val.y = color.y;
//		val.z = color.z;
//		val.w = 255;
//		surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
        }


        L = compwise_div(L, (L + Vec3(1.0f, 1.0f, 1.0f)));
        L = compwise_pow(L, Vec3(1.0f/2.2f, 1.0f/2.2f, 1.0f/2.2f));


        uchar4 val;
        val.x = static_cast<unsigned char>(L.x*255.0f);
        val.y = static_cast<unsigned char>(L.y*255.0f);
        val.z = static_cast<unsigned char>(L.z*255.0f);
        val.w = 255;
		surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
	}
}

__global__ void setupRandomState(curandState_t * state, uint64_t seed)
{
	int tid = gridDim.x*blockDim.x*(blockDim.y*blockIdx.y+threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}

__device__ float randUniform(curandState_t * state)
{
	int tid = gridDim.x*blockDim.x*(blockDim.y*blockIdx.y+threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	return curand_uniform(&state[tid]);
}

__device__ Vec3 randomInSphere(curandState_t * state)
{
	constexpr float pi = 3.14159265358979323846f;
	float pitch = randUniform(state)*pi*2.f;
	float yaw = randUniform(state)*pi*2.f;
	Vec3 res;
	res.x = cosf(yaw) * cosf(pitch);
	res.y = sinf(pitch);
	res.z = sinf(yaw) * cosf(pitch);
	res.normalize();
	float distance = sqrtf(randUniform(state));
	return res*distance;
}

__global__ void testFillFramebuffer(unsigned int w, unsigned int h, cudaSurfaceObject_t surface)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x<w && y<h)
	{
		uchar4 val = {255, 255, 0, 255};
		surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
	}
}

__global__ void renderStraight(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface)
{
	float x = w/2.0f;
	float y = h/2.0f;
	float ndc_x = (2.f*(float)x / (float)w)-1.f;
	float ndc_y = 1.f - (2.f*(float)y / (float)h);
	Vec4 vec{ndc_x, ndc_y, camNear, 1.f};
	vec = rayTransform*vec;
	Vec3 dir = Vec3{vec[0], vec[1], vec[2]};
	dir.normalize();

	Ray ray{camPos, dir.normalized()};

	Hit out;
	uchar4 val;
	if (scene->hit(ray, 0.1f, 50.0f, out))
	{
		val = {255, 0, 0, 255};
	}
}