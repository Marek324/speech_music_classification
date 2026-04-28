# CPU thread-scaling sweep

- date: 2026-04-26 10:33:20 UTC
- cpu: Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz
- thread counts: [1, 2, 4, 8]

| model | t1_ms | t2_ms | t4_ms | t8_ms | S(8) | amdahl_p |
|---|---:|---:|---:|---:|---:|---:|
| DT | 6.229 | 8.707 | 9.795 | 8.415 | 0.74× | -0.65 |
| GMM | 7.022 | 10.083 | 9.046 | 9.495 | 0.74× | -0.55 |
| SVM | 16.334 | 17.929 | 17.765 | 17.613 | 0.93× | -0.13 |
| SmallTCN | 14.923 | 12.663 | 14.755 | 12.224 | 1.22× | 0.17 |
| TCN | 14.237 | 11.770 | 12.227 | 6.026 | 2.36× | 0.40 |
| TCN+LSTM | 23.579 | 20.646 | 18.093 | 8.969 | 2.63× | 0.42 |

**S(N)** = t_f(1) / t_f(N) — observed parallel speedup at the
highest thread count tested. **amdahl_p** is the parallel fraction
fitted to all (N, S(N)) pairs: ≈0 means single-threaded by
construction, ≈1 means linear scaling, **negative** means anti-
scaling (S(N)<1 — threading overhead beats any parallel gain, as
happens with sklearn classifiers that aren't BLAS-bound during
predict). For a front-of-pipeline gate the user typically gives
the model 1 core, so multi-thread speedup is informational rather
than load-bearing.
