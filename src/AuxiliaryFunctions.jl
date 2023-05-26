R = pts -> ( z ->  map(y -> sqrt(z+eps()im - y |> Complex),pts) |> prod)

p(j,x) = x^j #basis polynomials for now

function cheby_int(bands)
    g = size(bands,1)-1
    cheby_coeffs = Array{Array{ComplexF64,1},2}(undef,g+1,g+2)
    for i = 1:g+1
        out_points = bands[1:end .!= i,:]
        for j = 1:g+2
            f = Fun( z -> p(j-1,z)/R(out_points)(z), bands[i,1]..bands[i,2])
            coeffs = coefficients(f)/√2
            coeffs[1] = coefficients(f)[1]
            cheby_coeffs[i,j] = coeffs
        end
    end
    cheby_coeffs
end

function cheby_gap(bands)
    g = size(bands,1)-1
    gaps = hcat(bands[1:end-1,2],bands[2:end,1])
    cheby_coeffs = Array{Array{ComplexF64,1},2}(undef,g,g+2)
    for i = 1:g
        out_points = vcat(bands[:,1][1:end .!= i+1],bands[:,2][1:end .!= i])
        for j = 1:g+2
            f = Fun( z -> p(j-1,z)/R(out_points)(z), gaps[i,1]..gaps[i,2])
            coeffs = coefficients(f)/√2
            coeffs[1] = coefficients(f)[1]
            cheby_coeffs[i,j] = coeffs
        end
    end
    cheby_coeffs
end

function g_coeffs(bands,gap_vals)
    g = size(bands,1)-1
    A = zeros(ComplexF64,g,g)
    b = zeros(ComplexF64,g)
    for i = 1:g
        out_points = vcat(bands[:,1][1:end .!= i+1],bands[:,2][1:end .!= i])
        for j = 0:g-1
            coeffs = gap_vals[i,j+1]
            c = R(bands[:])((bands[1,2]+bands[i+1,1])/2)/(R(out_points)((bands[i,2]+bands[i+1,1])/2)*abs(R([bands[i,2] bands[i+1,1]])((bands[i,2]+bands[i+1,1])/2))*pi)
            A[i,j+1] = coeffs[1]/c
        end
        coeffs = gap_vals[i,g+1]
        c = R(bands[:])((bands[1,2]+bands[i+1,1])/2)/(R(out_points)((bands[i,2]+bands[i+1,1])/2)*abs(R([bands[i,2] bands[i+1,1]])((bands[i,2]+bands[i+1,1])/2))*pi)
        b[i] = -coeffs[1]/c
    end
    #println("g matrix condition number:",cond(A))
    hvec = A\b
end

function compute_g_left(bands,hvec,int_vals)
    g_val = 0
    g = size(bands,1)-1
    z = bands[1,1]
    out_points = bands[1:end .!= 1,:]
    for j = 0:g-1
        coeffs = int_vals[1,j+1]
        n = length(coeffs)
        logtransT = zeros(ComplexF64,n)
        if n>1
            for k = 1:n-1
                logtransT[k+1] = (-1)^(k+1)*√2/k
            end
        end
        logtransT[1] = pi*im
        g_val += transpose(coeffs)*logtransT*hvec[j+1]
        #println(g_val)
    end
    coeffs = int_vals[1,g+1]
    n = length(coeffs)
    logtransT = zeros(ComplexF64,n)
    if n>1
        for k = 1:n-1
            logtransT[k+1] = (-1)^(k+1)*√2/k
        end
    end
    logtransT[1] = pi*im
    g_val += transpose(coeffs)*logtransT
    
    for i = 2:g+1
        out_points = bands[1:end .!= i,:]
        for j = 0:g-1
            coeffs = int_vals[i,j+1]
            n = length(coeffs)
            ChebyU = buildCheby(bands[i,1],bands[i,2],2)
            CauchyU = CauchyInterval(z,ChebyU,n)
            logtransT = zeros(ComplexF64,n)
            if n>1
                for k = 1:n-1
                    logtransT[k+1] = (π*im)*(bands[i,2]-bands[i,1])*CauchyU[k]/(√2*k)
                end
            end
            logtransT[1] = -log(J₊(iM(bands[i,1],bands[i,2])(z)))
            g_val += transpose(coeffs)*logtransT*hvec[j+1]
            #println(g_val)
        end
        coeffs = int_vals[i,g+1]
        n = length(coeffs)
        ChebyU = buildCheby(bands[i,1],bands[i,2],2)
        CauchyU = CauchyInterval(z,ChebyU,n)
        logtransT = zeros(ComplexF64,n)
        if n>1
            for k = 1:n-1
                logtransT[k+1] = (π*im)*(bands[i,2]-bands[i,1])*CauchyU[k]/(√2*k)
            end
        end
        logtransT[1] = -log(J₊(iM(bands[i,1],bands[i,2])(z)))
        g_val += transpose(coeffs)*logtransT
        #println(g_val)
    end
    g_val
end

function compute_g(z,bands,hvec,g_a,int_vals)
    g_val = 0
    g = size(bands,1)-1
    for i = 1:g+1
        out_points = bands[1:end .!= i,:]
        for j = 0:g-1
            coeffs = int_vals[i,j+1]
            n = length(coeffs)
            #println(n)
            ChebyU = buildCheby(bands[i,1],bands[i,2],2)
            CauchyU = CauchyInterval(z,ChebyU,n)
            logtransT = zeros(ComplexF64,n)
            if n>1
                for k = 1:n-1
                    logtransT[k+1] = (π*im)*(bands[i,2]-bands[i,1])*CauchyU[k]/(√2*k)
                end
            end
            logtransT[1] = -log(J₊(iM(bands[i,1],bands[i,2])(z)))
            g_val += transpose(coeffs)*logtransT*hvec[j+1]
        end
        # add monic term
        coeffs = int_vals[i,g+1]
        n = length(coeffs)
        #println(n)
        ChebyU = buildCheby(bands[i,1],bands[i,2],2)
        CauchyU = CauchyInterval(z,ChebyU,n)
        logtransT = zeros(ComplexF64,n)
        if n>1
            for k = 1:n-1
                logtransT[k+1] = (π*im)*(bands[i,2]-bands[i,1])*CauchyU[k]/(√2*k)
            end
        end
        logtransT[1] = -log(J₊(iM(bands[i,1],bands[i,2])(z)))
        g_val += transpose(coeffs)*logtransT
    end
    g_val -= g_a
end

struct gfunction
    bands
    hvec
    g_a
    int_vals
end

function (g::gfunction)(z)
    compute_g(z,g.bands,g.hvec,g.g_a,g.int_vals)
end

function correct_g(bands,hvec,g_a,int_vals)
    logcap, g₁ = 0,0
    g = size(bands,1)-1
    for i = 1:g+1
        out_points = bands[1:end .!= i,:]
        for j = 0:g-1
            coeffs = int_vals[i,j+1]
            capterm = log(4/(bands[i,2]-bands[i,1]))
            #println(capterm)
            g1term = -(bands[i,2]+bands[i,1])/2
            g1term2 = -(1/√2)*(bands[i,2]-bands[i,1])/2
            logcap += coeffs[1]*capterm*hvec[j+1]
            #println(cap)
            g₁ += coeffs[1]*g1term*hvec[j+1]+coeffs[2]*g1term2*hvec[j+1]
        end
        # add monic term
        coeffs = int_vals[i,g+1]
        capterm = log(4/(bands[i,2]-bands[i,1]))
        g1term = -(bands[i,2]+bands[i,1])/2
        g1term2 = -(1/√2)*(bands[i,2]-bands[i,1])/2
        logcap += coeffs[1]*capterm
        g₁ += coeffs[1]*g1term+coeffs[2]*g1term2
    end
    logcap -= g_a
    #logcap = real(logcap)
    cap = exp(logcap)
    (cap,g₁)
    #coefficients(f)
end

function h_coeffs_pre(bands,int_vals,gap_vals)
    # solve for coefficients to remove jump on gap
        gaps = hcat(bands[1:end-1,2],bands[2:end,1])
        g = size(bands,1)-1
        A = zeros(ComplexF64,g+1,g+1)
        B = zeros(ComplexF64,g+1,g)
        for i = 1:g+1
            out_points = bands[1:end .!= i,:]
            for j = 1:g+1
                coeffs = int_vals[i,j]
                c = R(bands[:])((bands[i,1]+bands[i,2])/2)/(R(out_points)((bands[i,1]+bands[i,2])/2)*abs(R(bands[i,:])((bands[i,1]+bands[i,2])/2))*pi)
                A[j,i] = coeffs[1]/c
            end
        end
        #println("h matrix condition number:",cond(A))    
        for i = 1:g
            bi = zeros(ComplexF64,g+1,1)
            out_points = vcat(bands[:,1][1:end .!= i+1],bands[:,2][1:end .!= i])
            for j = 1:g+1
                coeffs = gap_vals[i,j]
                c = R(bands[:])((gaps[i,1]+gaps[i,2])/2)/(R(out_points)((gaps[i,1]+gaps[i,2])/2)*abs(R(gaps[i,:])((gaps[i,1]+gaps[i,2])/2))*pi)
                bi[j] = coeffs[1]/c
            end
            B[:,i] = bi
        end
        (A,B)
end

function h_coeffs_post(A,B,Δ,n)
    b = -B*log.(exp.(n*Δ))   
    Avec = A\b
    if norm(imag(Avec))>1e-12
        println("Warning: computed h coefficients nonreal. Norm of imaginary part printed")
        norm(imag(Avec))
    end
    Avec = real(Avec)
end

function compute_h_pre(z,bands,int_vals,gap_vals)
    g = size(bands,1)-1
    h_vals = zeros(ComplexF64,2,g+1)
    for i = 1:g+1
        out_points = bands[1:end .!= i,:]
        coeffs = int_vals[i,1]
        nc = length(coeffs)
        #println(nc)
        ChebyT = buildCheby(bands[i,1],bands[i,2],1)
        CauchyT = CauchyInterval(z,ChebyT,nc-1)
        #global svde = CauchyT
        h_vals[1,i] = R(bands)(z)*(-π*im*CauchyT*coeffs)[1]
    end
    for i=1:g
        out_points = vcat(bands[:,1][1:end .!= i+1],bands[:,2][1:end .!= i])
        coeffs = gap_vals[i,1]
        nc = length(coeffs)
        #println(nc)
        ChebyT = buildCheby(bands[i,2],bands[i+1,1],1)
        CauchyT = CauchyInterval(z,ChebyT,nc-1)
        h_vals[2,i] = R(bands)(z)*(-π*im*CauchyT*coeffs)[1]        
    end
    h_vals
   # svde
end

struct hfunction_pre
    bands
    int_vals
    gap_vals
end

function (h_pre::hfunction_pre)(z)
    compute_h_pre(z,h_pre.bands,h_pre.int_vals,h_pre.gap_vals)
end

function compute_h_post(h_vals,Avec,Δ,n)
    sum(diag(h_vals*[Avec [log.(exp.(n*Δ)); 0]]))  
end

struct hfunction_post
    Avec
    Δ
    n
end

function (h_post::hfunction_post)(z)
    compute_h_post(z,h_post.Avec,h_post.Δ,h_post.n)
end

function correct_h_pre(bands,int_vals,gap_vals)
    gaps = hcat(bands[1:end-1,2],bands[2:end,1])
    g = size(bands,1)-1
    h_corr_vals = zeros(ComplexF64,2,g+1)
    for i = 1:g+1
        out_points = bands[1:end .!= i,:]
        coeffs = int_vals[i,g+2]
        c = R(bands[:])((bands[i,1]+bands[i,2])/2)/(R(out_points)((bands[i,1]+bands[i,2])/2)*abs(R(bands[i,:])((bands[i,1]+bands[i,2])/2))*pi)
        h_corr_vals[1,i] = (coeffs[1]/c)/(2π*im)
    end
    
    for i = 1:g
        out_points = vcat(bands[:,1][1:end .!= i+1],bands[:,2][1:end .!= i])
        coeffs = gap_vals[i,g+2]
        c = R(bands[:])((gaps[i,1]+gaps[i,2])/2)/(R(out_points)((gaps[i,1]+gaps[i,2])/2)*abs(R(gaps[i,:])((gaps[i,1]+gaps[i,2])/2))*pi)
        h_corr_vals[2,i] = (coeffs[1]/c)/(2π*im)
    end
h_corr_vals
end

function correct_h_post(h_corr_vals,Avec,Δ,n)
    h_correction = sum(diag(h_corr_vals*[Avec [log.(exp.(n*Δ)); 0]])) 
    σ₃ = [1 0; 0 -1]
    correction = h_correction*σ₃
end