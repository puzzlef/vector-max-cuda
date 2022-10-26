#pragma once
#include <vector>
#include "_main.hxx"
#include "max.hxx"

using std::vector;




template <bool POW2=false, class T>
MaxResult<T> maxCuda(const T *x, int N, const MaxOptions& o={}) {
  ASSERT(x);
  int RN = reduceSizeCu(N);
  size_t RT1 = RN * sizeof(T);
  size_t NT1 = N  * sizeof(T);
  ASSERT(RN<=N);
  T a   = T();
  T *r  = nullptr;
  T *rD = nullptr;
  T *xD = nullptr;
  // TRY( cudaProfilerStart() );
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&r, RT1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&rD, RT1) );
  TRY( cudaMalloc(&xD, NT1) );
  TRY( cudaMemcpy(xD, x.data(), NT1, cudaMemcpyHostToDevice) );
  float t = measureDuration([&] {
    maxCuW<POW2>(rD, xD, N);
    TRY( cudaMemcpy(r, rD, RT1, cudaMemcpyDeviceToHost) );
    a = maxValue(r, RN);
  }, o.repeat);
  return {a, t};
}

template <bool POW2=false, class T>
MaxResult<T> maxCuda(const vector<T>& x, const MaxOptions& o={}) {
  return maxCuda<POW2>(x.data(), x.size(), o);
}
