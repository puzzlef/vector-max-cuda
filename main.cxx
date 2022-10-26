#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include "src/main.hxx"

using namespace std;



#define TYPE float


void runSum(int N, int repeat) {
  vector<TYPE> x(N);
  for (int i=0; i<N; i++) {
    TYPE n = i+1;
    x[i] = 1/(n*n);
  }

  // Find Σx using a single thread.
  auto a1 = sumSeq(x, {repeat});
  printf("[%09.3f ms; %.0e elems.] [%f] sumSeq\n", a1.time, (double) N, a1.result);

  // Find Σx accelerated using OpenMP.
  auto a2 = sumOpenmp(x, {repeat});
  printf("[%09.3f ms; %.0e elems.] [%f] sumOpenmp\n", a2.time, (double) N, a2.result);
}


int main(int argc, char **argv) {
  int repeat = argc>1? stoi(argv[1]) : 5;
  for (int n=10000; n<=1000000000; n*=10)
    runSum(n, repeat);
  printf("\n");
  return 0;
}
