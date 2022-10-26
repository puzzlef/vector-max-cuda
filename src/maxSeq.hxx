#pragma once
#include <vector>
#include "max.hxx"

using std::vector;




template <class T>
MaxResult<T> maxSeq(const T *x, size_t N, const MaxOptions& o={}) {
  ASSERT(x);
  T a = T();
  float t = measureDuration([&]() {
    a = maxValue(x, N);
  }, o.repeat);
  return {a, t};
}

template <class T>
MaxResult<T> maxSeq(const vector<T>& x, const MaxOptions& o={}) {
  return maxSeq(x.data(), x.size(), o);
}
