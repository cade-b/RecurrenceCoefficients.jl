module RecurrenceCoefficients

using LinearAlgebra, ApproxFun
import Base: diff, *

include("CauchyInts.jl")
include("AuxiliaryFunctions.jl")

export get_coeffs, get_n_coeffs, get_coeffs_mixed, get_n_coeffs_mixed, get_coeffs1int, get_n_coeffs1int, pre_comp, pre_compU, pre_compV, pre_compW, get_coeffs_post, get_n_coeffs_post, get_n_coeffs_and_ints, get_n_coeffs_and_ints_mixed, get_n_coeffs_no_circ, get_n_coeffs_and_ints_no_circ, get_special_h, special_type

#Chebyshev T
function pre_comp(h, bands, nmat)
    # get all necessary ApproxFun coefficients at once
    int_vals = cheby_int(bands)
    gap_vals = cheby_gap(bands)

    hvec = g_coeffs(bands, gap_vals)
    g_a = compute_g_left(bands,hvec,int_vals)
    w(j) = z -> h(j)(z)/(√(bands[j,2]-z |> Complex)*√(z-bands[j,1] |> Complex))#z -> h(z)/(√(b-z |> Complex)*√(z-a |> Complex))
    global g = size(bands,1)-1
    global Δ = zeros(ComplexF64,g)
    for i = 1:g
        Δ[i]=compute_g((bands[i,2]+bands[i+1,1])/2+eps()im,bands,hvec,g_a,int_vals)-compute_g((bands[i,2]+bands[i+1,1])/2-eps()im,bands,hvec,g_a,int_vals)
    end

    global (cap,g₁)=correct_g(bands,hvec,g_a,int_vals)

    cc(j) = (bands[j,1]+bands[j,2])/2
    rr(j) = 1.25*(bands[j,2]-bands[j,1])/2
    gridmat = Array{Array{ComplexF64,1},2}(undef,g+1,2)

    for j = 1:g+1
        gridmat[j,1] = rr(j)*zgrid(nmat[j,1]).+cc(j).+eps()im #add perturbation to avoid branching issues
        gridmat[j,2] = M(bands[j,1],bands[j,2]).(Ugrid(nmat[j,2])) .|> Complex
    end

    ChebyTmat = Array{ChebyParams}(undef,g+1)
    ChebyUmat = Array{ChebyParams}(undef,g+1)
    for j = 1:g+1
        ChebyUmat[j] = buildCheby(bands[j,1],bands[j,2],2) 
        ChebyTmat[j] = buildCheby(bands[j,1],bands[j,2],1) 
    end

    global ntot = sum(sum(nmat))
    nptot(j) = sum(sum(nmat[1:j,:]))
    
    #build the portions of the matrix blocks not affected by the degree

    #build top left corner
    #build the first row of the block system
    global A₁₁ = zeros(ComplexF64,ntot,ntot); global rhs₁₁=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyUmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyUmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊= CauchyInterval(gridmat[j,2],ChebyUmat[j],nmat[j,2]-1;flag=1)
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyUmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyUmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        #println(maximum(abs.(CM₊-CM₋)))
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #global testguy=A₂
        #combine and build RHS
        A₁₁[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₁₁[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end

    global A₂₁ = zeros(ComplexF64,ntot,ntot); global rhs₂₁=zeros(ComplexF64,ntot,1)
    global Cmats1 = Array{Matrix{ComplexF64}}(undef,g+1)
    global Cmats2 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        #build portions of the first row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyTmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyTmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats1[j] = CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyTmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyTmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        Cmats2[j] = CM₋
    end

    #build bottom left corner
    #build the first row of the block system
    global A₁₂ = zeros(ComplexF64,ntot,ntot); global rhs₁₂=zeros(ComplexF64,ntot,1)
    global Cmats3 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyUmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyUmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = zeros(nmat[j,1])
        A₁ = -Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyUmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyUmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats3[j] = CM₋

        #combine and build RHS
        A₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2],:] = A₁
        rhs₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2]] = gn₁
    end

    #build bottom right corner
    #build the first row of the block system
    global A₂₂ = zeros(ComplexF64,ntot,ntot); global rhs₂₂=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyTmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyTmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊ = CauchyInterval(gridmat[j,2],ChebyTmat[j],nmat[j,2]-1;flag=1)
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyTmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyTmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #combine and build RHS
        A₂₂[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₂₂[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end
    
    #store values of g(z) on grid points on circles
    get_g(z) = compute_g(z,bands,hvec,g_a,int_vals)
    global g_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        g_vals[j] = map(z->get_g(z), gridmat[j,1])
    end
    #global g₀ = get_g(0)
    global gstuff = gfunction(bands,hvec,g_a,int_vals)
    
    #store values of modified weight function on intervals
    global w1_vals = Array{Vector{ComplexF64}}(undef,g+1)
    global w2_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w1_vals[j] = map(z->w(j)(z), gridmat[j,2])
        w2_vals[j] = map(z->-1/w(j)(z), gridmat[j,2])
    end
    
    #store values of modified weight function on circles
    m_w(j) = z -> imag(z)>0 ? -1/w(j)(z) : 1/w(j)(z)
    global w_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w_vals[j] = map(z->m_w(j)(z), gridmat[j,1])
    end
    
    #store terms for efficient computation of h(z)
    global (Ah_pre,Bh_pre) = h_coeffs_pre(bands,int_vals,gap_vals)
    global hcorr_pre = correct_h_pre(bands,int_vals,gap_vals)

    get_h_pre(z) = compute_h_pre(z, bands, int_vals, gap_vals)
    global h_vals_pre = Array{Vector{Matrix{ComplexF64}}}(undef,g+1)
    for j = 1:g+1
        h_vals_pre[j] = map(z->get_h_pre(z), gridmat[j,1])
    end
    global hstuff_pre = hfunction_pre(bands, int_vals, gap_vals)
end

function main_comp(bands,nmat,deg)
    rr(j) = 1.25*(bands[j,2]-bands[j,1])/2
    nptot(j) = sum(sum(nmat[1:j,:]))
    #perform the remaining computations
    Avec = h_coeffs_post(Ah_pre, Bh_pre, Δ, deg)

    #get values of h(z) on gridpoints on circles
    h_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        get_h(k) = compute_h_post(h_vals_pre[j][k], Avec, Δ, deg)
        h_vals[j] = map(k->get_h(k), 1:nmat[j,1])
    end
    #global h₀ = compute_h_post(h₀_pre, Avec, Δ, deg)
    global hstuff_post = hfunction_post(Avec, Δ, deg)
    
    jump_vals = Array{Vector{ComplexF64}}(undef,g+1)
    jump1_vals = Array{Vector{ComplexF64}}(undef,g+1)
    jump2_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        jump_vals[j] = w_vals[j].*exp.(2*h_vals[j]-2deg*g_vals[j])
        jump1_vals[j] = w1_vals[j]*exp(-Avec[j])
        jump2_vals[j] = w2_vals[j]*exp(Avec[j])
        
        #finish top right corner
        A₁ = -Diagonal(jump_vals[j])*Cmats1[j] 
        A₂ = -Diagonal(jump2_vals[j])*Cmats2[j]
        A₂₁[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₂₁[nptot(j-1)+1:nptot(j)] = [jump_vals[j]; jump2_vals[j]]
        
        #finish bottom left corner
        A₂ = -Diagonal(jump1_vals[j])*Cmats3[j]
        A₁₂[nptot(j)-nmat[j,2]+1:nptot(j),:] = A₂
        rhs₁₂[nptot(j)-nmat[j,2]+1:nptot(j)] = jump1_vals[j]
    end
    
    #build system
    A = [A₁₁ A₂₁; A₁₂ A₂₂]
    rhs = [rhs₁₁ rhs₂₁; rhs₁₂ rhs₂₂]
    coeffs = A\rhs
    #println(cond(A))
    
    #sort coefficients
    coeffmat₁₁ = Array{Array{ComplexF64,1},2}(undef,g+1,2)
    global coeffmat₁₂ = Array{Array{ComplexF64,1},2}(undef,g+1,2)
    coeffmat₂₁ = Array{Array{ComplexF64,1},2}(undef,g+1,2)
    coeffmat₂₂ = Array{Array{ComplexF64,1},2}(undef,g+1,2)

    for j=1:g+1
        coeffmat₁₁[j,1] = coeffs[nptot(j-1)+1:nptot(j)-nmat[j,2],1]
        coeffmat₁₁[j,2] = coeffs[nptot(j)-nmat[j,2]+1:nptot(j),1]

        coeffmat₁₂[j,1] = coeffs[ntot+nptot(j-1)+1:ntot+nptot(j)-nmat[j,2],1]
        coeffmat₁₂[j,2] = coeffs[ntot+nptot(j)-nmat[j,2]+1:ntot+nptot(j),1]

        coeffmat₂₁[j,1] = coeffs[nptot(j-1)+1:nptot(j)-nmat[j,2],2]
        coeffmat₂₁[j,2] = coeffs[nptot(j)-nmat[j,2]+1:nptot(j),2]

        coeffmat₂₂[j,1] = coeffs[ntot+nptot(j-1)+1:ntot+nptot(j)-nmat[j,2],2]
        coeffmat₂₂[j,2] = coeffs[ntot+nptot(j)-nmat[j,2]+1:ntot+nptot(j),2]
    end
    
    Y₁ = zeros(ComplexF64,2,2)
    for i = 1:g+1
        leading_jacobi = [coeffmat₁₁[i,2][1]*(im/2π) coeffmat₁₂[i,2][1]*(im/2π) ; coeffmat₂₁[i,2][1]*(im/2π)  coeffmat₂₂[i,2][1]*(im/2π)]
        leading_laurent = -[coeffmat₁₁[i,1][N₋(nmat[i,1])] coeffmat₁₂[i,1][N₋(nmat[i,1])]; coeffmat₂₁[i,1][N₋(nmat[i,1])] coeffmat₂₂[i,1][N₋(nmat[i,1])]]*rr(i)
        Y₁ += leading_jacobi+leading_laurent
    end
    h_correction = correct_h_post(hcorr_pre, Avec, Δ, deg)
    QMB = Y₁+h_correction
end

#for Chebyshev U weight
function pre_compU(h, bands, nmat)
    # get all necessary ApproxFun coefficients at once
    int_vals = cheby_int(bands)
    gap_vals = cheby_gap(bands)

    hvec = g_coeffs(bands, gap_vals)
    g_a = compute_g_left(bands,hvec,int_vals)
    w(j) = z -> h(j)(z)*(√(bands[j,2]-z |> Complex)*√(z-bands[j,1] |> Complex))#z -> h(z)/(√(b-z |> Complex)*√(z-a |> Complex))
    g = size(bands,1)-1
    global Δ = zeros(ComplexF64,g)
    for i = 1:g
        Δ[i]=compute_g((bands[i,2]+bands[i+1,1])/2+eps()im,bands,hvec,g_a,int_vals)-compute_g((bands[i,2]+bands[i+1,1])/2-eps()im,bands,hvec,g_a,int_vals)
    end

    global (cap,g₁)=correct_g(bands,hvec,g_a,int_vals)

    cc(j) = (bands[j,1]+bands[j,2])/2
    rr(j) = 1.25*(bands[j,2]-bands[j,1])/2
    global g = size(bands,1)-1
    gridmat = Array{Array{ComplexF64,1},2}(undef,g+1,2)

    for j = 1:g+1
        gridmat[j,1] = rr(j)*zgrid(nmat[j,1]).+cc(j).+eps()im
        gridmat[j,2] = M(bands[j,1],bands[j,2]).(Ugrid(nmat[j,2])) .|> Complex
    end

    ChebyTmat = Array{ChebyParams}(undef,g+1)
    ChebyUmat = Array{ChebyParams}(undef,g+1)
    for j = 1:g+1
        ChebyUmat[j] = buildCheby(bands[j,1],bands[j,2],2) 
        ChebyTmat[j] = buildCheby(bands[j,1],bands[j,2],1) 
    end

    global ntot = sum(sum(nmat))
    nptot(j) = sum(sum(nmat[1:j,:]))
    
    #build the portions of the matrix blocks not affected by the degree

    #build top left corner
    #build the first row of the block system
    global A₁₁ = zeros(ComplexF64,ntot,ntot); global rhs₁₁=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyTmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyTmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊= CauchyInterval(gridmat[j,2],ChebyTmat[j],nmat[j,2]-1;flag=1)
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyTmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyTmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        #println(maximum(abs.(CM₊-CM₋)))
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #global testguy=A₂
        #combine and build RHS
        A₁₁[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₁₁[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end

    global A₂₁ = zeros(ComplexF64,ntot,ntot); global rhs₂₁=zeros(ComplexF64,ntot,1)
    global Cmats1 = Array{Matrix{ComplexF64}}(undef,g+1)
    global Cmats2 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        #build portions of the first row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyUmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyUmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats1[j] = CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyUmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyUmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        Cmats2[j] = CM₋
    end

    #build bottom left corner
    #build the first row of the block system
    global A₁₂ = zeros(ComplexF64,ntot,ntot); global rhs₁₂=zeros(ComplexF64,ntot,1)
    global Cmats3 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyTmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyTmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = zeros(nmat[j,1])
        A₁ = -Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyTmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyTmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats3[j] = CM₋

        #combine and build RHS
        A₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2],:] = A₁
        rhs₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2]] = gn₁
    end

    #build bottom right corner
    #build the first row of the block system
    global A₂₂ = zeros(ComplexF64,ntot,ntot); global rhs₂₂=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyUmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyUmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊ = CauchyInterval(gridmat[j,2],ChebyUmat[j],nmat[j,2]-1;flag=1)
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyUmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyUmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #combine and build RHS
        A₂₂[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₂₂[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end
    
    #store values of g(z) on grid points on circles
    get_g(z) = compute_g(z,bands,hvec,g_a,int_vals)
    global g_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        g_vals[j] = map(z->get_g(z), gridmat[j,1])
    end
    #global g₀ = get_g(0)
    global gstuff = gfunction(bands,hvec,g_a,int_vals)
    
    #store values of modified weight function on intervals
    global w1_vals = Array{Vector{ComplexF64}}(undef,g+1)
    global w2_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w1_vals[j] = map(z->w(j)(z), gridmat[j,2])
        w2_vals[j] = map(z->-1/w(j)(z), gridmat[j,2])
    end
    
    #store values of modified weight function on circles
    m_w(j) = z -> imag(z)>0 ? -1/w(j)(z) : 1/w(j)(z)
    global w_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w_vals[j] = map(z->m_w(j)(z), gridmat[j,1])
    end
    
    #store terms for efficient computation of h(z)
    global (Ah_pre,Bh_pre) = h_coeffs_pre(bands,int_vals,gap_vals)
    global hcorr_pre = correct_h_pre(bands,int_vals,gap_vals)

    get_h_pre(z) = compute_h_pre(z, bands, int_vals, gap_vals)
    global h_vals_pre = Array{Vector{Matrix{ComplexF64}}}(undef,g+1)
    for j = 1:g+1
        h_vals_pre[j] = map(z->get_h_pre(z), gridmat[j,1])
    end
    #global h₀_pre = get_h_pre(0)
    global hstuff_pre = hfunction_pre(bands, int_vals, gap_vals)
end

function pre_compV(h, bands, nmat)
    # get all necessary ApproxFun coefficients at once
    int_vals = cheby_int(bands)
    gap_vals = cheby_gap(bands)

    hvec = g_coeffs(bands, gap_vals)
    g_a = compute_g_left(bands,hvec,int_vals)
    w(j) = z -> h(j)(z)*(√(z-bands[j,1] |> Complex)/√(bands[j,2]-z |> Complex))#z -> h(z)/(√(b-z |> Complex)*√(z-a |> Complex))
    g = size(bands,1)-1
    global Δ = zeros(ComplexF64,g)
    for i = 1:g
        Δ[i]=compute_g((bands[i,2]+bands[i+1,1])/2+eps()im,bands,hvec,g_a,int_vals)-compute_g((bands[i,2]+bands[i+1,1])/2-eps()im,bands,hvec,g_a,int_vals)
    end

    global (cap,g₁)=correct_g(bands,hvec,g_a,int_vals)

    cc(j) = (bands[j,1]+bands[j,2])/2
    rr(j) = 1.25*(bands[j,2]-bands[j,1])/2
    global g = size(bands,1)-1
    gridmat = Array{Array{ComplexF64,1},2}(undef,g+1,2)

    for j = 1:g+1
        gridmat[j,1] = rr(j)*zgrid(nmat[j,1]).+cc(j).+eps()im
        gridmat[j,2] = M(bands[j,1],bands[j,2]).(Ugrid(nmat[j,2])) .|> Complex
    end

    ChebyVmat = Array{ChebyParams}(undef,g+1)
    ChebyWmat = Array{ChebyParams}(undef,g+1)
    for j = 1:g+1
        ChebyVmat[j] = buildCheby(bands[j,1],bands[j,2],3) 
        ChebyWmat[j] = buildCheby(bands[j,1],bands[j,2],4) 
    end

    global ntot = sum(sum(nmat))
    nptot(j) = sum(sum(nmat[1:j,:]))
    
    #build the portions of the matrix blocks not affected by the degree

    #build top left corner
    #build the first row of the block system
    global A₁₁ = zeros(ComplexF64,ntot,ntot); global rhs₁₁=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyWmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyWmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊= CauchyInterval(gridmat[j,2],ChebyWmat[j],nmat[j,2]-1;flag=1)
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyWmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyWmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        #println(maximum(abs.(CM₊-CM₋)))
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #global testguy=A₂
        #combine and build RHS
        A₁₁[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₁₁[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end

    global A₂₁ = zeros(ComplexF64,ntot,ntot); global rhs₂₁=zeros(ComplexF64,ntot,1)
    global Cmats1 = Array{Matrix{ComplexF64}}(undef,g+1)
    global Cmats2 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        #build portions of the first row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyVmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyVmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats1[j] = CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyVmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyVmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        Cmats2[j] = CM₋
    end

    #build bottom left corner
    #build the first row of the block system
    global A₁₂ = zeros(ComplexF64,ntot,ntot); global rhs₁₂=zeros(ComplexF64,ntot,1)
    global Cmats3 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyWmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyWmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = zeros(nmat[j,1])
        A₁ = -Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyWmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyWmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats3[j] = CM₋

        #combine and build RHS
        A₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2],:] = A₁
        rhs₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2]] = gn₁
    end

    #build bottom right corner
    #build the first row of the block system
    global A₂₂ = zeros(ComplexF64,ntot,ntot); global rhs₂₂=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyVmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyVmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊ = CauchyInterval(gridmat[j,2],ChebyVmat[j],nmat[j,2]-1;flag=1)
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyVmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyVmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #combine and build RHS
        A₂₂[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₂₂[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end
    
    #store values of g(z) on grid points on circles
    get_g(z) = compute_g(z,bands,hvec,g_a,int_vals)
    global g_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        g_vals[j] = map(z->get_g(z), gridmat[j,1])
    end
    #global g₀ = get_g(0)
    global gstuff = gfunction(bands,hvec,g_a,int_vals)
    
    #store values of modified weight function on intervals
    global w1_vals = Array{Vector{ComplexF64}}(undef,g+1)
    global w2_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w1_vals[j] = map(z->w(j)(z), gridmat[j,2])
        w2_vals[j] = map(z->-1/w(j)(z), gridmat[j,2])
    end
    
    #store values of modified weight function on circles
    m_w(j) = z -> imag(z)>0 ? -1/w(j)(z) : 1/w(j)(z)
    global w_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w_vals[j] = map(z->m_w(j)(z), gridmat[j,1])
    end
    
    #store terms for efficient computation of h(z)
    global (Ah_pre,Bh_pre) = h_coeffs_pre(bands,int_vals,gap_vals)
    global hcorr_pre = correct_h_pre(bands,int_vals,gap_vals)

    get_h_pre(z) = compute_h_pre(z, bands, int_vals, gap_vals)
    global h_vals_pre = Array{Vector{Matrix{ComplexF64}}}(undef,g+1)
    for j = 1:g+1
        h_vals_pre[j] = map(z->get_h_pre(z), gridmat[j,1])
    end
    #global h₀_pre = get_h_pre(0)
    global hstuff_pre = hfunction_pre(bands, int_vals, gap_vals)
end

function pre_compW(h, bands, nmat)
    # get all necessary ApproxFun coefficients at once
    int_vals = cheby_int(bands)
    gap_vals = cheby_gap(bands)

    hvec = g_coeffs(bands, gap_vals)
    g_a = compute_g_left(bands,hvec,int_vals)
    w(j) = z -> h(j)(z)*(√(bands[j,2]-z |> Complex)/√(z-bands[j,1] |> Complex))#z -> h(z)/(√(b-z |> Complex)*√(z-a |> Complex))
    g = size(bands,1)-1
    global Δ = zeros(ComplexF64,g)
    for i = 1:g
        Δ[i]=compute_g((bands[i,2]+bands[i+1,1])/2+eps()im,bands,hvec,g_a,int_vals)-compute_g((bands[i,2]+bands[i+1,1])/2-eps()im,bands,hvec,g_a,int_vals)
    end

    global (cap,g₁)=correct_g(bands,hvec,g_a,int_vals)

    cc(j) = (bands[j,1]+bands[j,2])/2
    rr(j) = 1.25*(bands[j,2]-bands[j,1])/2
    global g = size(bands,1)-1
    gridmat = Array{Array{ComplexF64,1},2}(undef,g+1,2)

    for j = 1:g+1
        gridmat[j,1] = rr(j)*zgrid(nmat[j,1]).+cc(j).+eps()im
        gridmat[j,2] = M(bands[j,1],bands[j,2]).(Ugrid(nmat[j,2])) .|> Complex
    end

    ChebyVmat = Array{ChebyParams}(undef,g+1)
    ChebyWmat = Array{ChebyParams}(undef,g+1)
    for j = 1:g+1
        ChebyVmat[j] = buildCheby(bands[j,1],bands[j,2],3) 
        ChebyWmat[j] = buildCheby(bands[j,1],bands[j,2],4) 
    end

    global ntot = sum(sum(nmat))
    nptot(j) = sum(sum(nmat[1:j,:]))
    
    #build the portions of the matrix blocks not affected by the degree

    #build top left corner
    #build the first row of the block system
    global A₁₁ = zeros(ComplexF64,ntot,ntot); global rhs₁₁=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyVmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyVmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊= CauchyInterval(gridmat[j,2],ChebyVmat[j],nmat[j,2]-1;flag=1)
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyVmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyVmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        #println(maximum(abs.(CM₊-CM₋)))
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #global testguy=A₂
        #combine and build RHS
        A₁₁[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₁₁[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end

    global A₂₁ = zeros(ComplexF64,ntot,ntot); global rhs₂₁=zeros(ComplexF64,ntot,1)
    global Cmats1 = Array{Matrix{ComplexF64}}(undef,g+1)
    global Cmats2 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        #build portions of the first row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyWmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyWmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats1[j] = CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyWmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyWmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        Cmats2[j] = CM₋
    end

    #build bottom left corner
    #build the first row of the block system
    global A₁₂ = zeros(ComplexF64,ntot,ntot); global rhs₁₂=zeros(ComplexF64,ntot,1)
    global Cmats3 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyVmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyVmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = zeros(nmat[j,1])
        A₁ = -Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyVmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyVmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats3[j] = CM₋

        #combine and build RHS
        A₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2],:] = A₁
        rhs₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2]] = gn₁
    end

    #build bottom right corner
    #build the first row of the block system
    global A₂₂ = zeros(ComplexF64,ntot,ntot); global rhs₂₂=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyWmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyWmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊ = CauchyInterval(gridmat[j,2],ChebyWmat[j],nmat[j,2]-1;flag=1)
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyWmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyWmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #combine and build RHS
        A₂₂[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₂₂[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end
    
    #store values of g(z) on grid points on circles
    get_g(z) = compute_g(z,bands,hvec,g_a,int_vals)
    global g_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        g_vals[j] = map(z->get_g(z), gridmat[j,1])
    end
    global g₀ = get_g(0)
    
    #store values of modified weight function on intervals
    global w1_vals = Array{Vector{ComplexF64}}(undef,g+1)
    global w2_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w1_vals[j] = map(z->w(j)(z), gridmat[j,2])
        w2_vals[j] = map(z->-1/w(j)(z), gridmat[j,2])
    end
    
    #store values of modified weight function on circles
    m_w(j) = z -> imag(z)>0 ? -1/w(j)(z) : 1/w(j)(z)
    global w_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w_vals[j] = map(z->m_w(j)(z), gridmat[j,1])
    end
    
    #store terms for efficient computation of h(z)
    global (Ah_pre,Bh_pre) = h_coeffs_pre(bands,int_vals,gap_vals)
    global hcorr_pre = correct_h_pre(bands,int_vals,gap_vals)

    get_h_pre(z) = compute_h_pre(z, bands, int_vals, gap_vals)
    global h_vals_pre = Array{Vector{Matrix{ComplexF64}}}(undef,g+1)
    for j = 1:g+1
        h_vals_pre[j] = map(z->get_h_pre(z), gridmat[j,1])
    end
    global h₀_pre = get_h_pre(0)
end

function pre_comp_mixed(h, bands, nmat, typemat)
    # get all necessary ApproxFun coefficients at once
    int_vals = cheby_int(bands)
    gap_vals = cheby_gap(bands)

    hvec = g_coeffs(bands, gap_vals)
    g_a = compute_g_left(bands,hvec,int_vals)
    
    function w(j)
        if typemat[j] == 1 #T
            return z -> h(j)(z)/(√(bands[j,2]-z |> Complex)*√(z-bands[j,1] |> Complex))
        elseif typemat[j] == 2 #U
            return z -> h(j)(z)*(√(bands[j,2]-z |> Complex)*√(z-bands[j,1] |> Complex))
        elseif typemat[j] == 3 #V
            return z -> h(j)(z)*(√(z-bands[j,1] |> Complex)/√(bands[j,2]-z |> Complex))
        else #W
            return z -> h(j)(z)*(√(bands[j,2]-z |> Complex)/√(z-bands[j,1] |> Complex))
        end
    end
    
    g = size(bands,1)-1
    global Δ = zeros(ComplexF64,g)
    for i = 1:g
        Δ[i]=compute_g((bands[i,2]+bands[i+1,1])/2+eps()im,bands,hvec,g_a,int_vals)-compute_g((bands[i,2]+bands[i+1,1])/2-eps()im,bands,hvec,g_a,int_vals)
    end

    global (cap,g₁)=correct_g(bands,hvec,g_a,int_vals)

    cc(j) = (bands[j,1]+bands[j,2])/2
    rr(j) = 1.25*(bands[j,2]-bands[j,1])/2
    global g = size(bands,1)-1
    gridmat = Array{Array{ComplexF64,1},2}(undef,g+1,2)

    for j = 1:g+1
        gridmat[j,1] = rr(j)*zgrid(nmat[j,1]).+cc(j).+eps()im
        gridmat[j,2] = M(bands[j,1],bands[j,2]).(Ugrid(nmat[j,2])) .|> Complex
    end

    global ChebyAmat = Array{ChebyParams}(undef,g+1)
    ChebyBmat = Array{ChebyParams}(undef,g+1)
    for j = 1:g+1
        ChebyAmat[j] = buildCheby(bands[j,1],bands[j,2],typemat[j]) 
        ChebyBmat[j] = buildCheby(bands[j,1],bands[j,2],typemat[j]+2*(typemat[j]%2)-1) 
    end

    global ntot = sum(sum(nmat))
    nptot(j) = sum(sum(nmat[1:j,:]))
    
    #build the portions of the matrix blocks not affected by the degree

    #build top left corner
    #build the first row of the block system
    global A₁₁ = zeros(ComplexF64,ntot,ntot); global rhs₁₁=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyBmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyBmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊= CauchyInterval(gridmat[j,2],ChebyBmat[j],nmat[j,2]-1;flag=1)
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyBmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyBmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        #println(maximum(abs.(CM₊-CM₋)))
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #global testguy=A₂
        #combine and build RHS
        A₁₁[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₁₁[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end

    global A₂₁ = zeros(ComplexF64,ntot,ntot); global rhs₂₁=zeros(ComplexF64,ntot,1)
    global Cmats1 = Array{Matrix{ComplexF64}}(undef,g+1)
    global Cmats2 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        #build portions of the first row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyAmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyAmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats1[j] = CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋= CauchyInterval(gridmat[j,2],ChebyAmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyAmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        Cmats2[j] = CM₋
    end

    #build bottom left corner
    #build the first row of the block system
    global A₁₂ = zeros(ComplexF64,ntot,ntot); global rhs₁₂=zeros(ComplexF64,ntot,1)
    global Cmats3 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₋= CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyBmat[j],nmat[j,2]-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyBmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₁ = zeros(nmat[j,1])
        A₁ = -Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyBmat[j],nmat[j,2]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyBmat[k],nmat[k,2]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        Cmats3[j] = CM₋

        #combine and build RHS
        A₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2],:] = A₁
        rhs₁₂[nptot(j-1)+1:nptot(j)-nmat[j,2]] = gn₁
    end

    #build bottom right corner
    #build the first row of the block system
    global A₂₂ = zeros(ComplexF64,ntot,ntot); global rhs₂₂=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,1],ntot)
        CM₁₊ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=1)
        CM₁₋ = CauchyMat(gridmat[j,1],nmat[j,1],cc(j),rr(j);flag=-1)
        CM₂ = CauchyInterval(gridmat[j,1],ChebyAmat[j],nmat[j,2]-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₊
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]= CM₁₋
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,1],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,1],ChebyAmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]=CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]=CM₂
        end
        gn₁ = ones(nmat[j,1])
        A₁ = CM₊-Diagonal(gn₁)*CM₋
        #build the second row of the block system
        CM₊ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₋ = zeros(ComplexF64,nmat[j,2],ntot)
        CM₁ = CauchyMat(gridmat[j,2],nmat[j,1],cc(j),rr(j))
        CM₂₊ = CauchyInterval(gridmat[j,2],ChebyAmat[j],nmat[j,2]-1;flag=1)
        CM₂₋ = CauchyInterval(gridmat[j,2],ChebyAmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₋[:,nptot(j-1)+1:nptot(j)-nmat[j,2]]=CM₁
        CM₊[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₊
        CM₋[:,nptot(j)-nmat[j,2]+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₁ = CauchyMat(gridmat[j,2],nmat[k,1],cc(k),rr(k))
            CM₂ = CauchyInterval(gridmat[j,2],ChebyAmat[k],nmat[k,2]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₋[:,nptot(k-1)+1:nptot(k)-nmat[k,2]]= CM₁
            CM₊[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
            CM₋[:,nptot(k)-nmat[k,2]+1:nptot(k)]= CM₂
        end
        gn₂ = zeros(nmat[j,2])
        A₂ = CM₊-Diagonal(gn₂)*CM₋
        #combine and build RHS
        A₂₂[nptot(j-1)+1:nptot(j),:] = [A₁; A₂]
        rhs₁ = gn₁.-1
        rhs₂ = gn₂.-1
        rhs₂₂[nptot(j-1)+1:nptot(j)] = [rhs₁; rhs₂]
    end
    
    #store values of g(z) on grid points on circles
    get_g(z) = compute_g(z,bands,hvec,g_a,int_vals)
    global g_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        g_vals[j] = map(z->get_g(z), gridmat[j,1])
    end
    #global g₀ = get_g(0)
    global gstuff = gfunction(bands,hvec,g_a,int_vals)
    
    #store values of modified weight function on intervals
    global w1_vals = Array{Vector{ComplexF64}}(undef,g+1)
    global w2_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w1_vals[j] = map(z->w(j)(z), gridmat[j,2])
        w2_vals[j] = map(z->-1/w(j)(z), gridmat[j,2])
    end
    
    #store values of modified weight function on circles
    m_w(j) = z -> imag(z)>0 ? -1/w(j)(z) : 1/w(j)(z)
    global w_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w_vals[j] = map(z->m_w(j)(z), gridmat[j,1])
    end
    
    #store terms for efficient computation of h(z)
    global (Ah_pre,Bh_pre) = h_coeffs_pre(bands,int_vals,gap_vals)
    global hcorr_pre = correct_h_pre(bands,int_vals,gap_vals)

    get_h_pre(z) = compute_h_pre(z, bands, int_vals, gap_vals)
    global h_vals_pre = Array{Vector{Matrix{ComplexF64}}}(undef,g+1)
    for j = 1:g+1
        h_vals_pre[j] = map(z->get_h_pre(z), gridmat[j,1])
    end
    #global h₀_pre = get_h_pre(0)
    global hstuff_pre = hfunction_pre(bands, int_vals, gap_vals)
end

function get_coeffs(bands,n,kind::String="T",h::Function=j->(x->1);nmat=nothing)
    if nmat == nothing
        nmat = [120*ones(size(bands,1)) 20*ones(size(bands,1))] .|> Int128
    end

    if kind == "T"
        pre_comp(h, bands, nmat)
    elseif kind == "U"
        pre_compU(h, bands, nmat)
    elseif kind == "V"
        pre_compV(h, bands, nmat)
    elseif kind == "W"
        pre_compW(h, bands, nmat)
    end

    Y₁ = main_comp(bands,nmat,n)
    Y₁₊ = main_comp(bands,nmat,n+1)
    a = Y₁[1,1]-Y₁₊[1,1]-g₁
    b = √(Y₁₊[1,2]*Y₁₊[2,1])
    if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
        println("Warning: computed coefficient non-real. Imaginary parts printed")
        println(imag(a))
        println(imag(b))
    end
    a = real(a)
    b = real(b)
    (a,b)
end

function get_n_coeffs(bands,n,kind::String="T",h::Function=j->(x->1);nmat=nothing)
    if nmat == nothing
        nmat = [120*ones(size(bands,1)) 20*ones(size(bands,1))] .|> Int128
    end

    if kind == "T"
        pre_comp(h, bands, nmat)
    elseif kind == "U"
        pre_compU(h, bands, nmat)
    elseif kind == "V"
        pre_compV(h, bands, nmat)
    elseif kind == "W"
        pre_compW(h, bands, nmat)
    end

    Y₁ = main_comp(bands,nmat,0)
    avec = zeros(n+1); bvec = zeros(n+1)
    for j = 0:n
        Y₁₊ = main_comp(bands,nmat,j+1)
        a = Y₁[1,1]-Y₁₊[1,1]-g₁
        b = √(Y₁₊[1,2]*Y₁₊[2,1])
        if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
            println("Warning: computed coefficient non-real. Imaginary parts printed")
            println(imag(a))
            println(imag(b))
        end
        avec[j+1] = real(a)
        bvec[j+1] = real(b)
        Y₁ = Y₁₊
    end
    (avec,bvec)
end

function get_coeffs_mixed(bands,n,typemat,h::Function=j->(x->1);nmat=nothing)
    if nmat == nothing
        nmat = [120*ones(size(bands,1)) 20*ones(size(bands,1))] .|> Int128
    end

    pre_comp_mixed(h, bands, nmat, typemat)
    Y₁ = main_comp(bands,nmat,n)
    Y₁₊ = main_comp(bands,nmat,n+1)
    a = Y₁[1,1]-Y₁₊[1,1]-g₁
    b = √(Y₁₊[1,2]*Y₁₊[2,1])
    if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
        println("Warning: computed coefficient non-real. Imaginary parts printed")
        println(imag(a))
        println(imag(b))
    end
    a = real(a)
    b = real(b)
    (a,b)
end

function get_n_coeffs_mixed(bands,n,typemat,h::Function=j->(x->1);nmat=nothing)
    if nmat == nothing
        nmat = [120*ones(size(bands,1)) 20*ones(size(bands,1))] .|> Int128
    end

    pre_comp_mixed(h, bands, nmat, typemat)

    Y₁ = main_comp(bands,nmat,0)
    avec = zeros(n+1); bvec = zeros(n+1)
    for j = 0:n
        Y₁₊ = main_comp(bands,nmat,j+1)
        a = Y₁[1,1]-Y₁₊[1,1]-g₁
        b = √(Y₁₊[1,2]*Y₁₊[2,1])
        if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
            println("Warning: computed coefficient non-real. Imaginary parts printed")
            println(imag(a))
            println(imag(b))
        end
        avec[j+1] = real(a)
        bvec[j+1] = real(b)
        Y₁ = Y₁₊
    end
    (avec,bvec)
end

function get_coeffs_post(bands,n,nmat)
    Y₁ = main_comp(bands,nmat,n)
    Y₁₊ = main_comp(bands,nmat,n+1)
    a = Y₁[1,1]-Y₁₊[1,1]-g₁
    b = √(Y₁₊[1,2]*Y₁₊[2,1])
    if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
        println("Warning: computed coefficient non-real. Imaginary parts printed")
        println(imag(a))
        println(imag(b))
    end
    a = real(a)
    b = real(b)
    (a,b)
end

function get_n_coeffs_post(bands,n,nmat)

    Y₁ = main_comp(bands,nmat,0)
    avec = zeros(n+1); bvec = zeros(n+1)
    for j = 0:n
        Y₁₊ = main_comp(bands,nmat,j+1)
        a = Y₁[1,1]-Y₁₊[1,1]-g₁
        b = √(Y₁₊[1,2]*Y₁₊[2,1])
        if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
            println("Warning: computed coefficient non-real. Imaginary parts printed")
            println(imag(a))
            println(imag(b))
        end
        avec[j+1] = real(a)
        bvec[j+1] = real(b)
        Y₁ = Y₁₊
    end
    (avec,bvec)
end

function get_n_coeffs_and_ints(bands,n,eval_points,kind::String="T",h::Function=j->(x->1);nmat=nothing)
    if kind == "T"
        typemat = ones(size(bands,1)) .|> Int128
    elseif kind == "U"
        typemat = 2*ones(size(bands,1)) .|> Int128
    elseif kind == "V"
        typemat = 3*ones(size(bands,1)) .|> Int128
    elseif kind == "W"
        typemat = 4*ones(size(bands,1)) .|> Int128
    end

    get_n_coeffs_and_ints_mixed(bands,n,typemat,eval_points,h;nmat)
end

function get_n_coeffs_and_ints_mixed(bands,n,typemat,eval_points,h::Function=j->(x->1);nmat=nothing)
    if eval_points isa Number 
        eval_points = [eval_points]
    end

    if nmat == nothing
        nmat = [120*ones(size(bands,1)) 20*ones(size(bands,1))] .|> Int128
    end

    pre_comp_mixed(h, bands, nmat, typemat)
    Y₁ = main_comp(bands,nmat,0)
    cc(j) = (bands[j,1]+bands[j,2])/2
    rr(j) = 1.25*(bands[j,2]-bands[j,1])/2
    
    avec = zeros(n+1); bvec = zeros(n+1)
    ints = zeros(ComplexF64, n+1,length(eval_points))
    
    gz = gstuff.(eval_points)
    hz_pre = hstuff_pre.(eval_points)
    
    for (i,z) in enumerate(eval_points)
        S₀ = 0.
        for j = 1:g+1
            int_circ = CauchyEval(z,cc(j),rr(j),coeffmat₁₂[j,1])
            int_int =  CauchyInterval(z,ChebyAmat[j],nmat[j,2]-1)*coeffmat₁₂[j,2]
            S₀ += int_circ[1]+int_int[1]
        end 
        ints[1,i] = S₀
    end
        
    constprod = 1.
    
    for j = 0:n
        Y₁₊ = main_comp(bands,nmat,j+1)
        a = Y₁[1,1]-Y₁₊[1,1]-g₁
        b = √(Y₁₊[1,2]*Y₁₊[2,1])
        if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
            println("Warning: computed coefficient non-real. Imaginary parts printed")
            println(imag(a))
            println(imag(b))
        end
        avec[j+1] = real(a)
        bvec[j+1] = real(b)
        
        if j<n
            hz = hstuff_post.(hz_pre)
            constprod /= cap*bvec[j+1]
            for (i,z) in enumerate(eval_points)
                S₀ = 0.
                for j = 1:g+1
                    int_circ = CauchyEval(z,cc(j),rr(j),coeffmat₁₂[j,1])
                    int_int =  CauchyInterval(z,ChebyAmat[j],nmat[j,2]-1)*coeffmat₁₂[j,2]
                    S₀ += int_circ[1]+int_int[1]
                end
                ints[j+2,i] = constprod*S₀*exp(hz[i]-(j+1)*gz[i])
            end
        end
        
        Y₁ = Y₁₊
    end
    (avec,bvec,ints)
end

function get_special_h(bands)
    function special_h(j)
        if j == size(bands,1)
            return x -> -R(bands[1:end-1,1])(x)/R(bands[1:end-1,2])(x)
        else
            out_points = bands[1:end .!= j,:]
            return x -> (x-bands[end,2])*R(out_points[:,1])(x)/R(out_points[:,2])(x)
        end
    end
    special_h
end

function special_type(bands)
    typemat = 3*ones(size(bands,1)) .|> Int128
    typemat[end] = 2
    typemat
end

### functions that throw away the circles ###
#performs computation without circles; use h=special_h and typemat = special_type(bands) for accurate results
function pre_comp_mixed_no_circ(h, bands, nmat, typemat)
    # get all necessary ApproxFun coefficients at once
    int_vals = cheby_int(bands)
    gap_vals = cheby_gap(bands)

    hvec = g_coeffs(bands, gap_vals)
    g_a = compute_g_left(bands,hvec,int_vals)
    
    function w(j)
        if typemat[j] == 1 #T
            return z -> h(j)(z)/(√(bands[j,2]-z |> Complex)*√(z-bands[j,1] |> Complex))
        elseif typemat[j] == 2 #U
            return z -> h(j)(z)*(√(bands[j,2]-z |> Complex)*√(z-bands[j,1] |> Complex))
        elseif typemat[j] == 3 #V
            return z -> h(j)(z)*(√(z-bands[j,1] |> Complex)/√(bands[j,2]-z |> Complex))
        else #W
            return z -> h(j)(z)*(√(bands[j,2]-z |> Complex)/√(z-bands[j,1] |> Complex))
        end
    end
    
    global g = size(bands,1)-1
    global Δ = zeros(ComplexF64,g)
    for i = 1:g
        Δ[i]=compute_g((bands[i,2]+bands[i+1,1])/2+eps()im,bands,hvec,g_a,int_vals)-compute_g((bands[i,2]+bands[i+1,1])/2-eps()im,bands,hvec,g_a,int_vals)
    end

    global (cap,g₁)=correct_g(bands,hvec,g_a,int_vals)

    gridmat = Array{Array{ComplexF64,1}}(undef,g+1)
    for j = 1:g+1
        gridmat[j] = M(bands[j,1],bands[j,2]).(Ugrid(nmat[j])) .|> Complex
    end

    global ChebyAmat = Array{ChebyParams}(undef,g+1)
    ChebyBmat = Array{ChebyParams}(undef,g+1)
    for j = 1:g+1
        ChebyAmat[j] = buildCheby(bands[j,1],bands[j,2],typemat[j]) 
        ChebyBmat[j] = buildCheby(bands[j,1],bands[j,2],typemat[j]+2*(typemat[j]%2)-1) 
    end

    global ntot = sum(nmat)
    nptot(j) = sum(nmat[1:j])
    
    #build the portions of the matrix blocks not affected by the degree

    #build top left corner
    global A₁₁ = zeros(ComplexF64,ntot,ntot); global rhs₁₁=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j],ntot)
        #CM₋ = zeros(ComplexF64,nmat[j],ntot)
        CM₂₊= CauchyInterval(gridmat[j],ChebyBmat[j],nmat[j]-1;flag=1)
        #CM₂₋= CauchyInterval(gridmat[j],ChebyBmat[j],nmat[j,2]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)]= CM₂₊
        #CM₋[:,nptot(j-1)+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₂ = CauchyInterval(gridmat[j],ChebyBmat[k],nmat[k]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)]=CM₂
            #CM₋[:,nptot(k-1)+1:nptot(k)]=CM₂
        end
        #println(maximum(abs.(CM₊-CM₋)))
        gn₂ = zeros(nmat[j])
        A₂ = CM₊#-Diagonal(gn₂)*CM₋
        #global testguy=A₂
        #combine and build RHS
        A₁₁[nptot(j-1)+1:nptot(j),:] = A₂
        rhs₂ = gn₂.-1
        rhs₁₁[nptot(j-1)+1:nptot(j)] = rhs₂
    end

    global A₂₁ = zeros(ComplexF64,ntot,ntot); global rhs₂₁=zeros(ComplexF64,ntot,1)
    global Cmats2 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        CM₋ = zeros(ComplexF64,nmat[j],ntot)
        CM₂₋= CauchyInterval(gridmat[j],ChebyAmat[j],nmat[j]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₂ = CauchyInterval(gridmat[j],ChebyAmat[k],nmat[k]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)]=CM₂
        end
        Cmats2[j] = CM₋
    end

    #build bottom left corner
    global A₁₂ = zeros(ComplexF64,ntot,ntot); global rhs₁₂=zeros(ComplexF64,ntot,1)
    global Cmats3 = Array{Matrix{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        CM₋ = zeros(ComplexF64,nmat[j],ntot)
        CM₂₋ = CauchyInterval(gridmat[j],ChebyBmat[j],nmat[j]-1;flag=-1)
        CM₋[:,nptot(j-1)+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₂ = CauchyInterval(gridmat[j],ChebyBmat[k],nmat[k]-1)
            CM₋[:,nptot(k-1)+1:nptot(k)]= CM₂
        end
        Cmats3[j] = CM₋
    end

    #build bottom right corner
    global A₂₂ = zeros(ComplexF64,ntot,ntot); global rhs₂₂=zeros(ComplexF64,ntot,1)
    for j = 1:g+1
        CM₊ = zeros(ComplexF64,nmat[j],ntot)
        CM₋ = zeros(ComplexF64,nmat[j],ntot)
        CM₂₊ = CauchyInterval(gridmat[j],ChebyAmat[j],nmat[j]-1;flag=1)
        #CM₂₋ = CauchyInterval(gridmat[j],ChebyAmat[j],nmat[j]-1;flag=-1)
        CM₊[:,nptot(j-1)+1:nptot(j)]= CM₂₊
        #CM₋[:,nptot(j-1)+1:nptot(j)]= CM₂₋
        for k=(1:g+1)[1:end .!= j,:]
            CM₂ = CauchyInterval(gridmat[j],ChebyAmat[k],nmat[k]-1)
            CM₊[:,nptot(k-1)+1:nptot(k)]= CM₂
            #CM₋[:,nptot(k-1)+1:nptot(k)]= CM₂
        end
        gn₂ = zeros(nmat[j])
        A₂ = CM₊#-Diagonal(gn₂)*CM₋
        #combine and build RHS
        A₂₂[nptot(j-1)+1:nptot(j),:] = A₂
        rhs₂ = gn₂.-1
        rhs₂₂[nptot(j-1)+1:nptot(j)] = rhs₂
    end
    
    #store values of g(z) on grid points on circles
    get_g(z) = compute_g(z,bands,hvec,g_a,int_vals)
    global g_vals = Array{Vector{ComplexF64}}(undef,g+1)
    #global g₀ = get_g(0)
    global gstuff = gfunction(bands,hvec,g_a,int_vals)
    
    #store values of modified weight function on intervals
    global w1_vals = Array{Vector{ComplexF64}}(undef,g+1)
    global w2_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        w1_vals[j] = map(z->w(j)(z), gridmat[j])
        w2_vals[j] = map(z->-1/w(j)(z), gridmat[j])
    end
    
    #store terms for efficient computation of h(z)
    global (Ah_pre,Bh_pre) = h_coeffs_pre(bands,int_vals,gap_vals)
    global hcorr_pre = correct_h_pre(bands,int_vals,gap_vals)

    get_h_pre(z) = compute_h_pre(z, bands, int_vals, gap_vals)
    #global h₀_pre = get_h_pre(0)
    global hstuff_pre = hfunction_pre(bands, int_vals, gap_vals)
end

function main_comp_no_circ(bands,nmat,deg)
    nptot(j) = sum(nmat[1:j])
    #perform the remaining computations
    Avec = h_coeffs_post(Ah_pre, Bh_pre, Δ, deg)

    #global h₀ = compute_h_post(h₀_pre, Avec, Δ, deg)
    global hstuff_post = hfunction_post(Avec, Δ, deg)
    
    jump1_vals = Array{Vector{ComplexF64}}(undef,g+1)
    jump2_vals = Array{Vector{ComplexF64}}(undef,g+1)
    for j = 1:g+1
        jump1_vals[j] = w1_vals[j]*exp(-Avec[j])
        jump2_vals[j] = w2_vals[j]*exp(Avec[j])
        
        #finish top right corner 
        A₂ = -Diagonal(jump2_vals[j])*Cmats2[j]
        A₂₁[nptot(j-1)+1:nptot(j),:] = A₂
        rhs₂₁[nptot(j-1)+1:nptot(j)] = jump2_vals[j]
        
        #finish bottom left corner
        A₂ = -Diagonal(jump1_vals[j])*Cmats3[j]
        A₁₂[nptot(j-1)+1:nptot(j),:] = A₂
        rhs₁₂[nptot(j-1)+1:nptot(j)] = jump1_vals[j]
    end
    
    #build system
    A = [A₁₁ A₂₁; A₁₂ A₂₂]
    rhs = [rhs₁₁ rhs₂₁; rhs₁₂ rhs₂₂]
    coeffs = A\rhs
    #println(cond(A))
    
    #sort coefficients
    coeffmat₁₁ = Array{Array{ComplexF64,1}}(undef,g+1)
    global coeffmat₁₂ = Array{Array{ComplexF64,1}}(undef,g+1)
    coeffmat₂₁ = Array{Array{ComplexF64,1}}(undef,g+1)
    coeffmat₂₂ = Array{Array{ComplexF64,1}}(undef,g+1)

    for j=1:g+1
        coeffmat₁₁[j] = coeffs[nptot(j-1)+1:nptot(j),1]

        coeffmat₁₂[j] = coeffs[ntot+nptot(j-1)+1:ntot+nptot(j),1]

        coeffmat₂₁[j] = coeffs[nptot(j-1)+1:nptot(j),2]

        coeffmat₂₂[j] = coeffs[ntot+nptot(j-1)+1:ntot+nptot(j),2]
    end
    
    Y₁ = zeros(ComplexF64,2,2)
    for j = 1:g+1
        leading_jacobi = [coeffmat₁₁[j][1]*(im/2π) coeffmat₁₂[j][1]*(im/2π) ; coeffmat₂₁[j][1]*(im/2π)  coeffmat₂₂[j][1]*(im/2π)]     
        Y₁ += leading_jacobi
    end
    h_correction = correct_h_post(hcorr_pre, Avec, Δ, deg)
    QMB = Y₁+h_correction
end

function get_n_coeffs_no_circ(bands,n,typemat=nothing,h=nothing;nmat=nothing)
    if typemat == nothing
        typemat = special_type(bands)
    end

    if h == nothing
        h = get_special_h(bands)
    end

    if nmat == nothing
        nmat = 20*ones(size(bands,1)) .|> Int128
    end

    pre_comp_mixed_no_circ(h, bands, nmat, typemat)
    Y₁ = main_comp_no_circ(bands,nmat,0)
    avec = zeros(n+1); bvec = zeros(n+1)
    for j = 0:n
        Y₁₊ = main_comp_no_circ(bands,nmat,j+1)
        a = Y₁[1,1]-Y₁₊[1,1]-g₁
        b = √(Y₁₊[1,2]*Y₁₊[2,1])
        if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
            println("Warning: computed coefficient non-real. Imaginary parts printed")
            println(imag(a))
            println(imag(b))
        end
        avec[j+1] = real(a)
        bvec[j+1] = real(b)
        Y₁ = Y₁₊
    end
    (avec,bvec)
end

function get_n_coeffs_and_ints_no_circ(bands, n, eval_points,typemat=nothing,h=nothing;nmat=nothing)
    if typemat == nothing
        typemat = special_type(bands)
    end

    if h == nothing
        h = get_special_h(bands)
    end

    if nmat == nothing
        nmat = 20*ones(size(bands,1)) .|> Int128
    end

    pre_comp_mixed_no_circ(h, bands, nmat, typemat)
    Y₁ = main_comp_no_circ(bands,nmat,0)
    cc(j) = (bands[j,1]+bands[j,2])/2
    rr(j) = 1.25*(bands[j,2]-bands[j,1])/2
    
    avec = zeros(n+1); bvec = zeros(n+1)
    ints = zeros(ComplexF64, n+1,length(eval_points))
    
    gz = gstuff.(eval_points)
    hz_pre = hstuff_pre.(eval_points)
    
    for (i,z) in enumerate(eval_points)
        S₀ = 0.
        for j = 1:g+1
            int_int =  CauchyInterval(z,ChebyAmat[j],nmat[j]-1)*coeffmat₁₂[j]
            S₀ += int_int[1]
        end 
        ints[1,i] = S₀
    end
    
    constprod = 1.
    
    for j = 0:n
        Y₁₊ = main_comp_no_circ(bands,nmat,j+1)
        a = Y₁[1,1]-Y₁₊[1,1]-g₁
        b = √(Y₁₊[1,2]*Y₁₊[2,1])
        if abs(imag(a))>1e-12 || abs(imag(b))>1e-12
            println("Warning: computed coefficient non-real. Imaginary parts printed")
            println(imag(a))
            println(imag(b))
        end
        avec[j+1] = real(a)
        bvec[j+1] = real(b)
        
        if j<n
            hz = hstuff_post.(hz_pre)
            constprod /= cap*bvec[j+1]
            for (i,z) in enumerate(eval_points)
                S₀ = 0.
                for j = 1:g+1
                    int_int =  CauchyInterval(z,ChebyAmat[j],nmat[j]-1)*coeffmat₁₂[j]
                    S₀ += int_int[1]
                end
                ints[j+2,i] = constprod*S₀*exp(hz[i]-(j+1)*gz[i])
            end
        end
        
        Y₁ = Y₁₊
    end
    (avec,bvec,ints)
end

### single interval stuff ###
#configured for Chebyshev U
function SolveRHInf(h,deg,n₁,n₂,a,b) #can be further optimized, but works quick enough for now
    w(z) = h(z)*(√(b-z |> Complex)*√(z-a |> Complex))
    φ(z) = iM(a,b)(z)+√(iM(a,b)(z)+1 |> Complex)*√(iM(a,b)(z)-1 |> Complex)
    G₁(z) = [0 w(z); -1/w(z) 0]
    G₂(z) = imag(iM(a,b)(z))>0 ? [1 0; -(1/w(z))*φ(z)^(-2deg) 1] : [1 0; (1/w(z))*φ(z)^(-2deg) 1]
    
    global ChebyT = buildCheby(a,b,1) #Chebyshev T
    global ChebyU = buildCheby(a,b,2)  #Chebyshev U 
    
    global c, r = (b+a)/2 |> Complex, 1.25*(b-a)/2
    grid₁ = r*zgrid(n₁).+c .+eps()im
    grid₂ = M(a,b).(Ugrid(n₂)) .|> Complex

    #build top left corner
    #build the first row of the block system
    CM₁ = CauchyMat(grid₁,n₁,c,r;flag=1)
    CM₂ = CauchyInterval(grid₁,ChebyT,n₂-1)
    CM₃ = CauchyMat(grid₁,n₁,c,r;flag=-1)
    CM₊ = [CM₁ CM₂]
    CM₋ = [CM₃ CM₂]
    gn₁ = map(z -> G₂(z)[1,1], grid₁)
    A₁ = CM₊-Diagonal(gn₁)*CM₋
    #build the second row of the block system
    CM₁ = CauchyMat(grid₂,n₁,c,r)
    CM₂ = CauchyInterval(grid₂,ChebyT,n₂-1;flag=1)
    CM₃ = CauchyInterval(grid₂,ChebyT,n₂-1;flag=-1)
    gn₂ = map(z -> G₁(z)[1,1], grid₂)
    CM₊ = [CM₁ CM₂]
    CM₋ = [CM₁ CM₃]
    A₂ = CM₊-Diagonal(gn₂)*CM₋
    #combine and build RHS
    A₁₁ = [A₁; A₂]
    rhs₁ = gn₁.-1
    rhs₂ = gn₂.-1
    rhs₁₁ = [rhs₁; rhs₂]

    #build top right corner
    #build the first row of the block system
    CM₂ = CauchyInterval(grid₁,ChebyU,n₂-1)
    CM₃ = CauchyMat(grid₁,n₁,c,r;flag=-1)
    CM₋ = [CM₃ CM₂]
    gn₁ = map(z -> G₂(z)[2,1], grid₁)
    A₁ = -Diagonal(gn₁)*CM₋
    #build the second row of the block system
    CM₁ = CauchyMat(grid₂,n₁,c,r)
    CM₃ = CauchyInterval(grid₂,ChebyU,n₂-1;flag=-1)
    gn₂ = map(z -> G₁(z)[2,1], grid₂)
    CM₋ = [CM₁ CM₃]
    A₂ = -Diagonal(gn₂)*CM₋
    #combine and build RHS
    A₂₁ = [A₁; A₂]
    rhs₁ = gn₁
    rhs₂ = gn₂
    rhs₂₁ = [rhs₁; rhs₂]

    #build bottom left corner
    #build the first row of the block system
    CM₂ = CauchyInterval(grid₁,ChebyT,n₂-1)
    CM₃ = CauchyMat(grid₁,n₁,c,r;flag=-1)
    CM₋ = [CM₃ CM₂]
    gn₁ = map(z -> G₂(z)[1,2], grid₁)
    A₁ = -Diagonal(gn₁)*CM₋
    #build the second row of the block system
    CM₁ = CauchyMat(grid₂,n₁,c,r)
    CM₃ = CauchyInterval(grid₂,ChebyT,n₂-1;flag=-1)
    gn₂ = map(z -> G₁(z)[1,2], grid₂)
    CM₋ = [CM₁ CM₃]
    A₂ = -Diagonal(gn₂)*CM₋
    #combine and build RHS
    A₁₂ = [A₁; A₂]
    rhs₁ = gn₁
    rhs₂ = gn₂
    rhs₁₂ = [rhs₁; rhs₂]

    #build bottom right corner
    #build the first row of the block system
    CM₁ = CauchyMat(grid₁,n₁,c,r;flag=1)
    CM₂ = CauchyInterval(grid₁,ChebyU,n₂-1)
    CM₃ = CauchyMat(grid₁,n₁,c,r;flag=-1)
    CM₊ = [CM₁ CM₂]
    CM₋ = [CM₃ CM₂]
    gn₁ = map(z -> G₂(z)[2,2], grid₁)
    A₁ = CM₊-Diagonal(gn₁)*CM₋
    #build the second row of the block system
    CM₁ = CauchyMat(grid₂,n₁,c,r)
    CM₂ = CauchyInterval(grid₂,ChebyU,n₂-1;flag=1)
    CM₃ = CauchyInterval(grid₂,ChebyU,n₂-1;flag=-1)
    gn₂ = map(z -> G₁(z)[2,2], grid₂)
    CM₊ = [CM₁ CM₂]
    CM₋ = [CM₁ CM₃]
    A₂ = CM₊-Diagonal(gn₂)*CM₋
    #combine and build RHS
    A₂₂ = [A₁; A₂]
    rhs₁ = gn₁.-1
    rhs₂ = gn₂.-1
    rhs₂₂ = [rhs₁; rhs₂]

    #build system
    A = [A₁₁ A₂₁; A₁₂ A₂₂]
    rhs = [rhs₁₁ rhs₂₁; rhs₁₂ rhs₂₂]
    coeffs = A\rhs

    #organize coefficients
    global laurent_coeffs₁₁ = coeffs[1:n₁,1]
    global jacobi_coeffs₁₁ = coeffs[n₁+1:n₁+n₂,1]

    global laurent_coeffs₂₁ = coeffs[n₁+n₂+1:2n₁+n₂,1]
    global jacobi_coeffs₂₁ = coeffs[2n₁+n₂+1:end,1]

    global laurent_coeffs₁₂ = coeffs[1:n₁,2]
    global jacobi_coeffs₁₂ = coeffs[n₁+1:n₁+n₂,2]

    global laurent_coeffs₂₂ = coeffs[n₁+n₂+1:2n₁+n₂,2]
    global jacobi_coeffs₂₂ = coeffs[2n₁+n₂+1:end,2]
    
    #get leading order terms
    leading_jacobi = [jacobi_coeffs₁₁[1]*(im/2π) jacobi_coeffs₂₁[1]*(im/2π); jacobi_coeffs₁₂[1]*(im/2π) jacobi_coeffs₂₂[1]*(im/2π)]
    leading_laurent = -[laurent_coeffs₁₁[N₋(n₁)] laurent_coeffs₂₁[N₋(n₁)]; laurent_coeffs₁₂[N₋(n₁)] laurent_coeffs₂₂[N₋(n₁)]]*r
    Y₁ = leading_laurent+leading_jacobi
    #(leading_laurent,leading_jacobi)
end

function get_coeffs1int(h,n,a,b; num_points₁=120, num_points₂=16)
    c = (b+a)/2 
    Y₁ = SolveRHInf(h,n,num_points₁,num_points₂,a,b)
    Y₁₊ = SolveRHInf(h,n+1,num_points₁,num_points₂,a,b)
    aⱼ = Y₁[1,1]-Y₁₊[1,1]+c
    bⱼ = √(Y₁₊[1,2]*Y₁₊[2,1])
    if abs(imag(aⱼ))>1e-12 || abs(imag(bⱼ))>1e-12
        println("Warning: computed coefficient non-real")
    end
    (real(aⱼ),real(bⱼ))
end

function get_n_coeffs1int(h,n,a,b; num_points₁=120, num_points₂=20)
    c = (b+a)/2 
    Y₁ = SolveRHInf(h,0,num_points₁,num_points₂,a,b)
    avec = zeros(n+1); bvec = zeros(n+1)
    for j = 0:n
        Y₁₊ = SolveRHInf(h,j+1,num_points₁,num_points₂,a,b)
        aⱼ = Y₁[1,1]-Y₁₊[1,1]+c
        bⱼ = √(Y₁₊[1,2]*Y₁₊[2,1])
        if abs(imag(aⱼ))>1e-12 || abs(imag(bⱼ))>1e-12
            println("Warning: computed coefficient non-real. Imaginary parts printed")
            println(imag(aⱼ))
            println(imag(bⱼ))
        end
        avec[j+1] = real(aⱼ)
        bvec[j+1] = real(bⱼ)
        Y₁ = Y₁₊
    end
    (avec,bvec)
end

function get_n_coeffs1int(h,nstart,nend,a,b; num_points₁=120, num_points₂=20)
    c = (b+a)/2 
    Y₁ = SolveRHInf(h,nstart,num_points₁,num_points₂,a,b)
    avec = zeros(nend-nstart+1); bvec = zeros(nend-nstart+1)
    for j = nstart:nend
        Y₁₊ = SolveRHInf(h,j+1,num_points₁,num_points₂,a,b)
        aⱼ = Y₁[1,1]-Y₁₊[1,1]+c
        bⱼ = √(Y₁₊[1,2]*Y₁₊[2,1])
        if abs(imag(aⱼ))>1e-12 || abs(imag(bⱼ))>1e-12
            println("Warning: computed coefficient non-real. Imaginary parts printed")
            println(imag(aⱼ))
            println(imag(bⱼ))
        end
        avec[j-nstart+1] = real(aⱼ)
        bvec[j-nstart+1] = real(bⱼ)
        Y₁ = Y₁₊
    end
    (avec,bvec)
end

end # module
