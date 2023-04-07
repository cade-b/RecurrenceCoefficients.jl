using RecurrenceCoefficients
using Test

@testset "core tests" begin
    bands = [-3.0 -2.0; 2.0 3.0]
    (avec,bvec) = get_n_coeffs(bands,50)

    tol = 1e-12
    @test maximum(abs.(avec))<tol
    @test abs(bvec[50]-0.5)<tol
    @test abs(bvec[51]-2.5)<tol
end