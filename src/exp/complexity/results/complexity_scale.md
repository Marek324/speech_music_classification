# CPU thread-scaling sweep

- date: 2026-04-29 13:14:38 UTC
- cpu: Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz
- thread counts: [1, 2, 4, 8]

| model | t1_ms | t2_ms | t4_ms | t8_ms | S(8) | amdahl_p |
|---|---:|---:|---:|---:|---:|---:|
| DT | 4.323 | 4.475 | 4.489 | 4.288 | 1.01× | -0.04 |
| GMM | 4.000 | 4.111 | 4.194 | 4.267 | 0.94× | -0.07 |
| SVM | 7.478 | 7.497 | 8.286 | 7.354 | 1.02× | -0.04 |
| SmallTCN | 7.249 | 5.872 | 5.339 | 7.173 | 1.01× | 0.25 |
| SmallerTCN | 4.729 | 3.788 | 3.692 | 3.883 | 1.22× | 0.30 |
| TCN | 7.346 | 5.795 | 5.248 | 6.178 | 1.19× | 0.33 |
| TCN+LSTM | 12.241 | 8.802 | 7.503 | 8.686 | 1.41× | 0.47 |

**S(N)** = t_f(1) / t_f(N) — observed parallel speedup at the
highest thread count tested. **amdahl_p** is the parallel fraction
fitted to all (N, S(N)) pairs: ≈0 means single-threaded by
construction, ≈1 means linear scaling, **negative** means anti-
scaling (S(N)<1 — threading overhead beats any parallel gain, as
happens with sklearn classifiers that aren't BLAS-bound during
predict). For a front-of-pipeline gate the user typically gives
the model 1 core, so multi-thread speedup is informational rather
than load-bearing.
