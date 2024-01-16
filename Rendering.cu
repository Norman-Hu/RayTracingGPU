#include <Rendering.cuh>
#include <surface_indirect_functions.h>
#include <cstdio>

#define PI 3.14159265358979323846f
#define INV_PI 0.31830988618379067154f
#define INV_2PI 0.15915494309189533577f
#define INV_4PI 0.07957747154594766788f

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

__host__ __device__ Vec3 sampleHemisphereAroundNormal(float u, float v, const Vec3 & normal)
{
    Vec3 tangent;
    if (fabs(normal.x) > fabs(normal.y))
        tangent = (Vec3(-normal.z, 0.0f, normal.x)).normalized();
    else
        tangent = (Vec3(0.0f, normal.z, -normal.y)).normalized();

    Vec3 bitangent = -Vec3::cross(normal, tangent);

    // Transform the sampled point from local space to world space
    float z = u;
    float r = sqrtf(1.0f - z * z);
    float phi = 2.0f * PI * v;

    // Sampled point in local coordinates
    Vec3 localPoint = {r * cosf(phi), r * sinf(phi), z};

    // Transform local coordinates to world coordinates
    Vec3 worldPoint = localPoint.x * tangent + localPoint.y * bitangent + localPoint.z * normal;

    return worldPoint;
}

__host__ __device__ Vec3 sampleUniformSphere(float u, float v)
{
    float z = 1.0f - 2.0f * u;
    float r = sqrtf(1.0f - z*z);
    float phi = 2.0f * PI * v;
    return {r * cosf(phi), r * sinf(phi), z};
}

__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__global__ void render(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface, curandState_t * randState, bool sampleLights, int sampleCount)
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
        Vec3 res;
        Ray cur_ray = ray;
        Vec3 cur_attenuation = Vec3(1.0f,1.0f,1.0f);
        for (int i=0; i<5; ++i)
        {
            Hit rec;
            if (scene->hit(cur_ray, 0.001f, 100.0f, rec))
            {
                PBRMaterial & mat = scene->materials[rec.materialId];

                Vec3 target = rec.p + rec.normal + sampleUniformSphere(curand_uniform(randState+tid),
                                                                       curand_uniform(randState+tid));
                Ray scattered = Ray{rec.p, (target-rec.p).normalized()};
                Vec3 attenuation = mat.albedo;
                cur_attenuation = compwise_mul(attenuation, cur_attenuation);
                cur_ray = scattered;

                if (scene->lightCount > 0)
                {
                    Light *light = scene->lights[static_cast<int>(curand_uniform(randState + tid) * ((float)scene->lightCount-0.00001f))];
                    LightSamples s = light->getSamples(randState);
                    for (int i = 0; i < s.size; ++i) {
                        Vec3 lightDir = s.samples[i] - rec.p;
                        Hit lightHit;
                        if (!scene->hit({rec.p, lightDir.normalized()}, 0.001f, lightDir.length(), lightHit)) {
                            res += compwise_mul(cur_attenuation, light->color);
                        }
                    }
                }
            }
            else
            {
//                float t = 0.5f*(cur_ray.direction.y + 1.0f);
//                Vec3 c = (1.0f-t)*Vec3(1.0f, 1.0f, 1.0f) + t*Vec3(0.5f, 0.7f, 1.0f);
//                res = compwise_mul(cur_attenuation, c);
                break;
            }
        }

        Vec3 L = res;

//        L = compwise_div(L, (L + Vec3(1.0f, 1.0f, 1.0f)));
        L /= (L + Vec3(1.0f, 1.0f, 1.0f)).max();
        L = compwise_pow(L, Vec3(1.0f/2.2f, 1.0f/2.2f, 1.0f/2.2f));

        L*=255.0f;
        if (sampleCount == 0)
        {
            uchar4 val;
            val.x = static_cast<unsigned char>(L.x);
            val.y = static_cast<unsigned char>(L.y);
            val.z = static_cast<unsigned char>(L.z);
            val.w = 255;
            surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
        }
        else
        {
            // recover current state and average out
            uchar4 currentUchar = surf2Dread<uchar4>(surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);

            Vec3 newVal(currentUchar.x, currentUchar.y, currentUchar.z);
            newVal += (L-newVal)/((float)sampleCount+1);

            uchar4 val;
            val.x = static_cast<unsigned char>(newVal.x);
            val.y = static_cast<unsigned char>(newVal.y);
            val.z = static_cast<unsigned char>(newVal.z);
            val.w = 255;
		    surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
        }
	}
}

__global__ void renderSimple(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface, curandState_t * randState, bool sampleLights, int sampleCount)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = gridDim.x*blockDim.x*(blockDim.y*blockIdx.y+threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    if (x<w && y<h) {
        // Find ray
        float ndc_x = (2.f * (float) x / (float) w) - 1.f;
        float ndc_y = 1.f - (2.f * (float) y / (float) h);
        Vec4 vec{ndc_x, ndc_y, camNear, 1.f};
        vec = rayTransform * vec;
        Vec3 dir = Vec3{vec[0], vec[1], vec[2]};
        dir.normalize();
        Ray ray{camPos, dir.normalized()};

        uchar4 val{0, 0, 0, 255};
        Hit hitInfo;
        bool hit = scene->hit(ray, 0.001f, 100.0f, hitInfo);
        if (hit) {
            if (hitInfo.t > 10.0f)
                hitInfo.t = 10.0f;
            hitInfo.t /= 10.0f;
            hitInfo.t = 1.0f - hitInfo.t;
            val.x = hitInfo.t * 255.0f;
            val.y = val.x;
            val.z = val.x;
        }
        surf2Dwrite<uchar4>(val, surface, (int) sizeof(uchar4) * x, y, cudaBoundaryModeClamp);
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