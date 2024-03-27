#include <utility> 

template <typename T>
void CUDATranspose(T** device_data, std::pair<int, int>& size);

template <typename T>
void CUDAMultiply(T** device_data_a, T** device_data_b, T**& device_data_multiplied, std::pair<int, int> size_a, std::pair<int, int> size_b);

template <typename T>
void CUDAscalarMultiply(T** device_data, float multiplicand, T**& device_data_multiplied, std::pair<int, int> size);

template <typename T>
void CUDAscalarAdd(T** device_data, T to_add, T**& device_data_added, std::pair<int, int> size);

template <typename T>
void CUDAdivide(T** device_data_a, T** device_data_b, T**& device_data_divided, std::pair<int, int> size);

template <typename T>
void CUDAadd(T** device_data_a, T** device_data_b, T**& device_data_added, std::pair<int, int> size);

template <typename T>
void CUDAelemMultiply(T** device_data_a, T** device_data_b, T**& device_data_multiplied, std::pair<int, int> size);

template <typename T>
void CUDAflatten(T** device_data, T*& device_data_flattened, std::pair<int, int> size);

template <typename T>
void CUDAreshape(T** device_data, T**& device_data_reshaped, std::pair<int, int> oldSize, std::pair<int, int> newSize);

template <typename T>
void CUDAconvertFloat(T** device_data, std::pair<int, int> size);

template <typename T>
void CUDAsqrt(T** device_data, float**& device_data_sqrt, std::pair<int, int> size);