#ifndef Alloc_H
#define Alloc_H
#include <cstdio>
#include <cstdlib> 
#include <cuda_runtime.h>

__host__ __device__
inline long get_idx(long v, long w, long x, long y, long z, long stride_w, long stride_x, long stride_y, long stride_z)
{
    return stride_x * stride_y * stride_z * w + stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long w, long x, long y, long z, long stride_x, long stride_y, long stride_z)
{
    return stride_x * stride_y * stride_z * w + stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long x, long y, long z, long stride_y, long stride_z)
{
    return stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long x, long y, long s1)
{
    return x + (y * s1);
}


template < class type >
inline type *newArr1(size_t sz1)
{
  type *arr = new type[sz1];
  return arr;
}

template < class type >
inline type **newArr2(size_t sz1, size_t sz2)
{
  type **arr = new type*[sz1]; // new type *[sz1];
  type *ptr = newArr1<type>(sz1*sz2);
  for (size_t i = 0; i < sz1; i++)
  {
    arr[i] = ptr;
    ptr += sz2;
  }
  return arr;
}

template < class type >
inline type ***newArr3(size_t sz1, size_t sz2, size_t sz3)
{
  type ***arr = new type**[sz1]; // new type **[sz1];
  type **ptr = newArr2<type>(sz1*sz2, sz3);
  for (size_t i = 0; i < sz1; i++)
  {
    arr[i] = ptr;
    ptr += sz2;
  }
  return arr;
}

template <class type>
inline type ****newArr4(size_t sz1, size_t sz2, size_t sz3, size_t sz4)
{
  type ****arr = new type***[sz1]; //(new type ***[sz1]);
  type ***ptr = newArr3<type>(sz1*sz2, sz3, sz4);
  for (size_t i = 0; i < sz1; i++) {
    arr[i] = ptr;
    ptr += sz2;
  }
  return arr;
}

// build chained pointer hierarchy for pre-existing bottom level
//

/* Build chained pointer hierachy for pre-existing bottom level                        *
 * Provide a pointer to a contig. 1D memory region which was already allocated in "in" *
 * The function returns a pointer chain to which allows subscript access (x[i][j])     */
template <class type>
inline type *****newArr5(type **in, size_t sz1, size_t sz2, size_t sz3, size_t sz4, size_t sz5)
{
  *in = newArr1<type>(sz1*sz2*sz3*sz4*sz5);

  type*****arr = newArr4<type*>(sz1,sz2,sz3,sz4);
  type**arr2 = ***arr;
  type *ptr = *in;
  size_t szarr2 = sz1*sz2*sz3*sz4;
  for(size_t i=0;i<szarr2;i++) {
    arr2[i] = ptr;
    ptr += sz5;
  }
  return arr;
}

template <class type>
inline type ****newArr4(type **in, size_t sz1, size_t sz2, size_t sz3, size_t sz4)
{
  *in = newArr1<type>(sz1*sz2*sz3*sz4);

  type****arr = newArr3<type*>(sz1,sz2,sz3);
  type**arr2 = **arr;
  type *ptr = *in;
  size_t szarr2 = sz1*sz2*sz3;
  for(size_t i=0;i<szarr2;i++) {
    arr2[i] = ptr;
    ptr += sz4;
  }
  return arr;
}

template <class type>
inline type ***newArr3(type **in, size_t sz1, size_t sz2, size_t sz3)
{
  *in = newArr1<type>(sz1*sz2*sz3);

  type***arr = newArr2<type*>(sz1,sz2);
  type**arr2 = *arr;
  type *ptr = *in;
  size_t szarr2 = sz1*sz2;
  for(size_t i=0;i<szarr2;i++) {
    arr2[i] = ptr;
    ptr += sz3;
  }
  return arr;
}

template <class type>
inline type **newArr2(type **in, size_t sz1, size_t sz2)
{
  *in = newArr1<type>(sz1*sz2);
  type**arr = newArr1<type*>(sz1);
  type *ptr = *in;
  for(size_t i=0;i<sz1;i++) {
    arr[i] = ptr;
    ptr += sz2;
  }
  return arr;
}

// methods to deallocate arrays
//
template < class type > inline void delArray1(type * arr)
{ delete[](arr); }
template < class type > inline void delArray2(type ** arr)
{ delArray1(arr[0]); delete[](arr); }
template < class type > inline void delArray3(type *** arr)
{ delArray2(arr[0]); delete[](arr); }
template < class type > inline void delArray4(type **** arr)
{ delArray3(arr[0]); delete[](arr); }
//
// versions with dummy dimensions (for backwards compatibility)
//
template <class type> inline void delArr1(type * arr)
{ delArray1<type>(arr); }
template <class type> inline void delArr2(type ** arr, size_t sz1)
{ delArray2<type>(arr); }
template <class type> inline void delArr3(type *** arr, size_t sz1, size_t sz2)
{ delArray3<type>(arr); }
template <class type> inline void delArr4(type **** arr, size_t sz1, size_t sz2, size_t sz3)
{ delArray4<type>(arr); }

#define newArr1(type, sz1) newArr1<type>(sz1)
#define newArr(type,sz1,sz2) newArr2<type>(sz1, sz2)
#define newArr2(type, sz1, sz2) newArr2<type>(sz1, sz2)
#define newArr3(type, sz1, sz2, sz3) newArr3<type>(sz1, sz2, sz3)
#define newArr4(type, sz1, sz2, sz3, sz4) newArr4<type>(sz1, sz2, sz3, sz4)

// ------------------------------------------------------------
// Pinned host-memory allocators (for faster H<->D transfers)
// These create the same pointer-chain layout as newArr3(in,...)
// but allocate the flat buffer with cudaMallocHost.
// ------------------------------------------------------------

static inline void cudaCheckAlloc(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
    std::abort();
  }
}

// 3D array with pinned flat storage + chained pointers
template <class type>
inline type ***newArr3Pinned(type **flat_out, size_t sz1, size_t sz2, size_t sz3)
{
  // allocate pinned contiguous block
  type *flat = nullptr;
  cudaCheckAlloc(cudaMallocHost((void**)&flat, sz1 * sz2 * sz3 * sizeof(type)), "cudaMallocHost newArr3Pinned");
  *flat_out = flat;

  // build pointer chain (same structure as newArr3(in, ...))
  type ***arr = newArr2<type*>(sz1, sz2);
  type **arr2 = *arr;
  type *ptr = flat;
  size_t szarr2 = sz1 * sz2;
  for (size_t i = 0; i < szarr2; i++) {
    arr2[i] = ptr;
    ptr += sz3;
  }
  return arr;
}

// Free 3D array allocated with newArr3Pinned
template <class type>
inline void delArr3Pinned(type ***arr, type *flat)
{
  if (!arr) return;
  // Free pointer chain (it was allocated with newArr2<type*> -> delArray2 works)
  delArray2<type*>(arr);
  // Free pinned flat buffer
  if (flat) cudaCheckAlloc(cudaFreeHost(flat), "cudaFreeHost delArr3Pinned");
}

#endif
