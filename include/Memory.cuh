#ifndef MEMORY_H
#define MEMORY_H

template <typename T>
__host__ __device__ constexpr T exchange(T & ref, T newVal)
{
	T tmp = ref;
	ref = newVal;
	return tmp;
}

template <typename T>
__host__ __device__ constexpr void swap(T & ref1, T & ref2)
{
	T tmp = ref1;
	ref1 = ref2;
	ref2 = tmp;
}


#endif // MEMORY_H
