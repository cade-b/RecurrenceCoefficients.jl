M = (a,b) ->  (x -> (b-a)/2*(x .+ (b+a)/(b-a)))
iM = (a,b) -> (x -> 2/(b-a)*(x .- (b+a)/2))

mgrid = (n,L) -> -L .+ 2*L*(0:n-1)/n
zgrid = n -> exp.(1im*mgrid(n,pi))

Ugrid = n -> cos.( (2*(1:n) .- 1)/(2*n) * pi)

N₋ = m -> convert(Int64,floor(m/2))
N₊ = m -> convert(Int64,floor((m-1)/2))

struct ChebyParams
    a::Float64
    b::Float64
    kind::Int
end

function buildCheby(kind)
    ChebyParams(-1,1,kind)
end

function buildCheby(a,b,kind)
    ChebyParams(a,b,kind)
end

J₊(z) = z-√(z-1 |> Complex)*√(z+1 |> Complex) #inverse Joukowsky map

function ChebyTIntExact(z,N,a,b)
    C = zeros(ComplexF64,N+1)
    C[1] = √2*im/(2*pi*√(z-b |> Complex)*√(z-a |> Complex))
    for k = 1:N
        C[k+1] = C[k]*J₊(iM(a,b)(z))
    end
    C[1] /= √2
    C
end

function ChebyUIntExact(z,N,a,b)
    C = zeros(ComplexF64,N+1)
    C[1] = im*J₊(iM(a,b)(z))/π*(2/(b-a))
    for k = 1:N
        C[k+1] = C[k]*J₊(iM(a,b)(z))
    end
    C
end

function ChebyVIntExact(z,N,a,b)
    C = zeros(ComplexF64,N+1)
    C[1] = (im/2π)*(-1 + sqrt((z-a)/(z-b) |> Complex))*(2/(b-a))
    for k = 1:N
        C[k+1] = C[k]*J₊(iM(a,b)(z))
    end
    C
end

function ChebyWIntExact(z,N,a,b)
    C = zeros(ComplexF64,N+1)
    C[1] = (im/2π)*(1 - sqrt((z-b)/(z-a) |> Complex))*(2/(b-a))
    for k = 1:N
        C[k+1] = C[k]*J₊(iM(a,b)(z))
    end
    C
end

function CauchyIntervalVec(z, X::ChebyParams, N)
    if X.kind == 1
        C = ChebyTIntExact(z,N,X.a,X.b)
    elseif X.kind == 2
        C = ChebyUIntExact(z,N,X.a,X.b)
    elseif X.kind == 3
        C = ChebyVIntExact(z,N,X.a,X.b)
    elseif X.kind == 4
        C = ChebyWIntExact(z,N,X.a,X.b)
    end
transpose(C) |> Array
end

function CauchyInterval(Z, X::ChebyParams, N; flag = 0)
    if flag == 0
        x = map(z -> CauchyIntervalVec(z, X::ChebyParams, N), Z)
        if length(Z)>1
            x = reduce(vcat,x)
        end
    elseif flag == 1
        x = CauchyInterval(Z.+eps()im, X, N)
    elseif flag == -1
        x = CauchyInterval(Z.-eps()im, X, N)
    end
    x
end

# build vectors of Cauchy matrices at single collocation point
function CauchyVec₊(y,n)
    vec = zeros(ComplexF64,1,n)
    for j = 0:N₊(n)
        vec[j+N₋(n)+1] = y^j
    end
    vec
end

function CauchyVec₋(y,n)
    vec = zeros(ComplexF64,1,n)
    for j = 1:N₋(n)
        vec[j] = -y^(-N₋(n)-1+j)
    end
    vec
end

function CauchyVec(y,n;flag=0)
    if flag==0
        if abs(y)<1
            return CauchyVec₊(y,n)
        elseif abs(y)>1
            return CauchyVec₋(y,n)
        else
            println("Need flag; 1 for + and -1 for -")
        end
    elseif flag == 1
        return CauchyVec₊(y,n)
    elseif flag == -1
        return CauchyVec₋(y,n)
    end
end

# build Cauchy vector for different circular contours
function CauchyVec₊(y,n,c,r)
    CauchyVec₊((y-c)/r,n)
end

function CauchyVec₋(y,n,c,r)
    CauchyVec₊((y-c)/r,n)
end

function CauchyVec(y,n,c,r;flag=0)
    CauchyVec((y-c)/r,n;flag=flag)
end

function CauchyMat(Y,n;flag=0)
    x = map(y -> CauchyVec(y,n;flag=flag),Y)
    if length(Y)>1
        x = reduce(vcat,x)
    end
    x
end

function CauchyMat(Y,n,c,r;flag=0)
    CauchyMat((Y.-c)/r,n;flag=flag)
end