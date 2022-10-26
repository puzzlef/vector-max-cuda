Performance of *sequential* vs *OpenMP-based* **vector element sum**.

This experiment was for comparing the performance between:
1. Find `Σx` using a single thread (**sequential**).
2. Find `Σx` accelerated using **OpenMP**.

Here `x` is a floating-point vector. Both approaches were attempted on a number
of vector sizes, running each approach 5 times per size to get a good time
measure. Note that neither approach makes use of *SIMD instructions* which are
available on all modern hardware. While it might seem that **OpenMP** method
would be a clear winner, the results indicate it is dependent upon the workload.
Results indicate that **from 10⁵ elements, OpenMP** **approach performs better**
than sequential.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets].

<br>

```bash
$ g++ -O3 -fopenmp main.cxx
$ ./a.out

# [00000.053 ms; 1e+04 elems.] [1.644725] sumSeq
# [00000.446 ms; 1e+04 elems.] [1.644834] sumOpenmp
# [00000.347 ms; 1e+05 elems.] [1.644725] sumSeq
# [00000.067 ms; 1e+05 elems.] [1.644919] sumOpenmp
# [00001.852 ms; 1e+06 elems.] [1.644725] sumSeq
# [00000.090 ms; 1e+06 elems.] [1.644756] sumOpenmp
# [00014.484 ms; 1e+07 elems.] [1.644725] sumSeq
# [00001.646 ms; 1e+07 elems.] [1.644728] sumOpenmp
# [00148.799 ms; 1e+08 elems.] [1.644725] sumSeq
# [00008.172 ms; 1e+08 elems.] [1.644726] sumOpenmp
# [01660.869 ms; 1e+09 elems.] [1.644725] sumSeq
# [00078.691 ms; 1e+09 elems.] [1.644725] sumOpenmp
```

[![](https://i.imgur.com/gdJQIVJ.png)][sheetp]
[![](https://i.imgur.com/4VKbI6Q.png)][sheetp]

<br>
<br>


## References

- [Parallel for loop in openmp](https://stackoverflow.com/a/11773714/1413259)
- [What's the difference between “static” and “dynamic” schedule in OpenMP?](https://stackoverflow.com/a/10852852/1413259)

<br>
<br>

[![](https://i.imgur.com/Va4qBPD.jpg)](https://www.youtube.com/watch?v=0XTLuFpuAtE)

[charts]: https://photos.app.goo.gl/ruBBeXTz6vEATFQc6
[sheets]: https://docs.google.com/spreadsheets/d/1ZBp9TFxisTn_8_7_X_2HSPt_SbmP16gUW6_fiMtr8rc/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vTPFQlVfelJvgStWlpUnmAWcTZZXpYbL4APW0nc0djGLIGJdWABXx6S2QLkrxHbct_jnXJ51JlB7cv7/pubhtml
