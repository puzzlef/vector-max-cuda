#include <cmath>
#include <random>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include "src/main.hxx"

using namespace std;


#define TYPE int




template <class T=int, class R>
vector<T> randomValues(size_t N, R& rnd) {
  uniform_int_distribution<T> dis(T(0), T(1000000));
  vector<T> a(N);
  for (size_t i=0; i<N; ++i)
    a[i] = dis(rnd);
  return a;
}


template <class T>
void runBatch(const vector<T>& x, int repeat) {
  size_t N = x.size();
  // Find max() using a single thread.
  auto a1 = maxSeq(x, {repeat});
  printf("[%09.3f ms; %.0e elems.] [%d] maxSeq\n",      a1.time, (double) N, a1.result);
  // Find max() accelerated using CUDA (not pow2).
  auto a2 = maxCuda(x, {repeat});
  printf("[%09.3f ms; %.0e elems.] [%d] maxCuda\n",     a2.time, (double) N, a2.result);
  ASSERT(a2.result==a1.result);
  // Find max() accelerated using CUDA (pow2).
  auto a3 = maxCuda<true>(x, {repeat});
  printf("[%09.3f ms; %.0e elems.] [%d] maxCudaPow2\n", a3.time, (double) N, a3.result);
  ASSERT(a3.result==a1.result);
}


void runExperiment(int repeat) {
  using T = TYPE;
  random_device dev;
  default_random_engine rnd(dev());
  for (size_t N=1000; N<=1000000000; N*=10) {
    for (int n=0; n<repeat; ++n) {
      vector<T> x = randomValues<T>(N, rnd);
      runBatch(x, repeat);
    }
  }
}


int main(int argc, char **argv) {
  int repeat = argc>1? stoi(argv[1]) : 5;
  runExperiment(repeat);
  printf("\n");
  return 0;
}
