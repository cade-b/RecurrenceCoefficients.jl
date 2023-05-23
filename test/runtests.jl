using RecurrenceCoefficients
using Test

@testset "core tests" begin
    tol = 1e-12

    #check symmetric intervals
    bands = [-3.0 -2.0; 2.0 3.0]
    (avec,bvec) = get_n_coeffs(bands,50)
    @test maximum(abs.(avec))<tol
    @test abs(bvec[50]-0.5)<tol
    @test abs(bvec[51]-2.5)<tol

    #check against Householder
    bands = [0.1 1.1; 2.0 3.0; 3.5 4.0]
    (a,b) = get_coeffs(bands,1,"V")
    @test abs(a-2.146157169447241)<tol
    @test abs(b-0.9021505173702458)<tol

    bands = [-3.2 -2.2; 0.1 1.1; 2.0 3.0; 3.5 4.0]
    typemat = [1 2 3 4]
    (avec,bvec) = get_n_coeffs_mixed(bands,2,typemat)
    at = [-0.18333333333333274; 0.5015878159235749; 0.6807273164837916]
    bt = [2.7814963998219078; 1.0954803914792746; 2.1352466569622113]
    @test maximum(abs.(avec-at))<tol
    @test maximum(abs.(bvec-bt))<tol

    #check pre/post comp gives the same thing
    bands = [-3.6 -2.4; 0.1 0.5; 1.5 2.3]
    (a1,b1) = get_n_coeffs(bands,50,"W")
    nmat = [120*ones(size(bands,1)) 20*ones(size(bands,1))] .|> Int128
    typemat = 4*ones(size(bands,1)) .|> Int128
    pre_comp(j->x->1,bands,nmat,typemat)
    (a2,b2) = get_n_coeffs_post(bands,50,nmat)
    @test maximum(abs.(a1-a2)) == 0.
    @test maximum(abs.(b1-b2)) == 0.

    #check Cauchy integral
    bands = [-4.5 -3.0; -2.0 -1.0; 2.0 3.0]
    (avec,bvec,ints)=get_n_coeffs_and_ints(bands,50,0.0)
    @test abs(ints[4]-0.05126582226964149im)<tol

    #check throwing away circle
    bands = [-3.3 -2.0; -1.0 0.0; 2.0 3.0];
    (avect,bvect) = get_n_coeffs_mixed(bands,50,special_type(bands),get_special_h(bands))
    (avecn,bvecn) = get_n_coeffs_no_circ(bands,50)
    (avecn1,bvecn1) = get_n_coeffs_no_circ(bands,25,50)
    @test maximum(abs.(avect-avecn))<tol
    @test maximum(abs.(bvect-bvecn))<tol
    @test maximum(abs.(avecn1-avecn[26:end]))<tol
    @test maximum(abs.(bvecn1-bvecn[26:end]))<tol
end