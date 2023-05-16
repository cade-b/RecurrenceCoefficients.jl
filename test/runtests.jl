using RecurrenceCoefficients
using Test

@testset "core tests" begin
    bands = [-3.0 -2.0; 2.0 3.0]
    (avec,bvec) = get_n_coeffs(bands,50)

    tol = 1e-12
    @test maximum(abs.(avec))<tol
    @test abs(bvec[50]-0.5)<tol
    @test abs(bvec[51]-2.5)<tol

    bands = [-3.6 -2.4; 0.1 0.5; 1.5 2.3]
    (a1,b1) = get_n_coeffs(bands,50,"V")
    nmat = [120*ones(size(bands,1)) 20*ones(size(bands,1))] .|> Int128
    pre_compV(j->x->1,bands,nmat)
    (a2,b2) = get_n_coeffs_post(bands,50,nmat)
    @test maximum(abs.(a1-a2)) == 0.
    @test maximum(abs.(b1-b2)) == 0.
end