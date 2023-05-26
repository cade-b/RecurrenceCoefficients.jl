# RecurrenceCoefficients.jl

Compute recurrence coefficients for Chebyshev polynomials defined on multiple intervals.

This respository contains the code for: [arxiv.org/abs/2302.12930](https://arxiv.org/abs/2302.12930)

A basic example usage follows.
```
bands = [-3.0 -2.0; 2.0 3.0] #intervals of support
(a,b) = get_coeffs(bands,5) #computes coefficients a₅,b₅
(avec,bvec) = get_n_coeffs(bands,5) #computes coefficients a₀,…,a₅,b₀,…,b₅
```
Optional arguments include the choice of Chebyshev variant, a weight function perturbation, and the number of collocation points used. Example usage of these optional arguments follows.
```
bands = [-3.0 -2.0; 2.0 3.0]
h(j) = x->exp(0.01j*x)
nmat = [100*ones(size(bands,1)) 15*ones(size(bands,1))] .|> Int128
(avec,bvec) = get_n_coeffs(bands,5,"V",h;nmat) #computes coefficients a₀,…,a₅,b₀,…,b₅
```
To use different Chebyshev variants on each interval, different functions must be called. The following example uses Chebyshev- $T$, $U$, $V$, and $W$ weights in that order.
```
bands = [-3.2 -2.2; 0.1 1.1; 2.0 3.0; 3.5 4.0]
typemat = [1 2 3 4]
(a,b) = get_coeffs_mixed(bands,5,typemat) #computes coefficients a₅,b₅
(avec,bvec) = get_n_coeffs_mixed(bands,5,typemat) #computes coefficients a₀,…,a₅,b₀,…,b₅
```


[![CI](https://github.com/cade-b/RecurrenceCoefficients.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/cade-b/RecurrenceCoefficients.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cade-b/RecurrenceCoefficients.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/cade-b/RecurrenceCoefficients.jl)
