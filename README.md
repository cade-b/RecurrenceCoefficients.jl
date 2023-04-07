# RecurrenceCoefficients.jl

Compute recurrence coefficients for Chebyshev polynomials defined on multiple intervals.

This respository contains the code for: [arxiv.org/abs/2302.12930](https://arxiv.org/abs/2302.12930)

A basic example usage follows.
```
bands = [-3.0 -2.0; 2.0 3.0] #intervals of support
(a,b) = get_coeffs(bands,5) #computes coefficients a₅,b₅
(avec,bvec) = get_n_coeffs(bands,5) #computes coefficients a₀,…,a₅,b₀,…,b₅
```

