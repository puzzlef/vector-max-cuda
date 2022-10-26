#pragma once




struct MaxOptions {
  int repeat;

  MaxOptions(int repeat=1) :
  repeat(repeat) {}
};


template <class T>
struct MaxResult {
  T     result;
  float time;

  MaxResult(T result, float time) :
  result(result), time(time) {}
};
