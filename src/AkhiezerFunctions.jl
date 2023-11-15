#***BEGIN PROLOGUE  DRF
#***PURPOSE  Compute the incomplete or complete elliptic integral of the
#            1st kind.  For X, Y, and Z non-negative and at most one of
#            them zero, RF(X,Y,Z) = Integral from zero to infinity of
#                                -1/2     -1/2     -1/2
#                      (1/2)(t+X)    (t+Y)    (t+Z)    dt.
#            If X, Y or Z is zero, the integral is complete.
#***LIBRARY   SLATEC
#***CATEGORY  C14
#***TYPE      DOUBLE PRECISION (RF-S, DRF-D)
#***KEYWORDS  COMPLETE ELLIPTIC INTEGRAL, DUPLICATION THEOREM,
#             INCOMPLETE ELLIPTIC INTEGRAL, INTEGRAL OF THE FIRST KIND,
#             TAYLOR SERIES
#***AUTHOR  Carlson, B. C.
#             Ames Laboratory-DOE
#             Iowa State University
#             Ames, IA  50011
#           Notis, E. M.
#             Ames Laboratory-DOE
#             Iowa State University
#             Ames, IA  50011
#           Pexton, R. L.
#             Lawrence Livermore National Laboratory
#             Livermore, CA  94550
const D1MACH1 = floatmin(Float64)
const D1MACH2 = floatmax(Float64)
const D1MACH3 = eps(Float64)/2
const D1MACH4 = eps(Float64)
const D1MACH5 = log10(2.)
function DRF(X::ComplexF64, Y::ComplexF64, Z::Float64)
    ERRTOL = (4.0*D1MACH3)^(1.0/6.0)
    LOLIM  = 5.0 * D1MACH1
    UPLIM  = D1MACH2/5.0
    C1 = 1.0/24.0
    C2 = 3.0/44.0
    C3 = 1.0/14.0

    ans = 0.0
    #=if min(X,Y,Z) < 0.0
        return ans, 1
    end

    if max(X,Y,Z) > UPLIM
        return ans, 3
    end

    if min(X+Y,X+Z,Y+Z) < LOLIM
        return ans, 2
end=#

    XN = X
    YN = Y
    ZN = Z
    MU = 0.
    XNDEV = 0.
    YNDEV = 0.
    ZNDEV = 0.

    while true
        MU = (XN+YN+ZN)/3.0
        XNDEV = 2.0 - (MU+XN)/MU
        YNDEV = 2.0 - (MU+YN)/MU
        ZNDEV = 2.0 - (MU+ZN)/MU
        EPSLON = max(abs(XNDEV),abs(YNDEV),abs(ZNDEV))
        if (EPSLON < ERRTOL) break end
        XNROOT = sqrt(XN)
        YNROOT = sqrt(YN)
        ZNROOT = sqrt(ZN)
        LAMDA = XNROOT*(YNROOT+ZNROOT) + YNROOT*ZNROOT
        XN = (XN+LAMDA)*0.250
        YN = (YN+LAMDA)*0.250
        ZN = (ZN+LAMDA)*0.250
    end

    E2 = XNDEV*YNDEV - ZNDEV*ZNDEV
    E3 = XNDEV*YNDEV*ZNDEV
    S  = 1.0 + (C1*E2-0.10-C2*E3)*E2 + C3*E3
    ans = S/sqrt(MU)

    return ans, 0
end

function rawF(sinphi::ComplexF64, m::Float64)
    if abs(sinphi) == 1. && m == 1. return sign(sinphi)*Inf end
    sinphi2 = sinphi^2
    drf,ierr = DRF(1. - sinphi2, 1. - m*sinphi2, 1.)
    @assert ierr == 0
    sinphi*drf
end

function Elliptic.F(phi::ComplexF64, m::Float64)
    if isnan(phi) || isnan(m) return NaN end
    if m < 0. || m > 1. throw(DomainError(m, "argument m not in [0,1]")) end
    if abs(real(phi)) > pi/2
        # Abramowitz & Stegun (17.4.3)
        phi2 = real(phi) + pi/2
        return 2*fld(phi2,pi)*Elliptic.K(m) - rawF(cos(mod(phi2,pi)+imag(phi)im), m)
    end
    rawF(sin(phi), m)
end

function θ₁(z,q)
    res = 0
    τ=log(q)/(pi*im)
    #error bound from  Labrande (2015)
    maxiter = ceil(sqrt(66/(pi*imag(τ)*log(2,exp(1)))))+1 |> Int
    for n = 0:maxiter
        res += 2*(-1)^n*q^((n+1/2)^2)*sin(z*(2n+1))
    end
    res
end

function dθ₁(z,q)
    res = 0
    τ=log(q)/(pi*im)
    #error bound from  Labrande (2015)
    maxiter = ceil(sqrt(66/(pi*imag(τ)*log(2,exp(1)))))+1 |> Int
    for n = 0:maxiter
        res += (2n+1)*2*(-1)^n*q^((n+1/2)^2)*cos(z*(2n+1))
    end
    res
end

function θ₄(z,q)
    res = 1
    τ=log(q)/(pi*im)
    #error bound from  Labrande (2015)
    maxiter = ceil(sqrt(66/(pi*imag(τ)*log(2,exp(1)))))+1 |> Int
    for n = 1:maxiter
        res += 2*(-1)^n*q^(n^2)*cos(z*2n)
    end
    res
end

function dθ₄(z,q)
    res = 0
    τ=log(q)/(pi*im)
    #error bound from  Labrande (2015)
    maxiter = ceil(sqrt(66/(pi*imag(τ)*log(2,exp(1)))))+1 |> Int
    for n = 1:maxiter
        res += -4n*(-1)^n*q^(n^2)*sin(z*2n)
    end
    res
end

d2sn(x,m) = -m*(Elliptic.Jacobi.cn(x,m))^2*Elliptic.Jacobi.sn(x,m)-Elliptic.Jacobi.sn(x,m)*(Elliptic.Jacobi.dn(x,m)^2)

function CIp(bands,n,x)
    α = iM(bands[1,1],bands[2,2])(bands[1,2])
    β = iM(bands[1,1],bands[2,2])(bands[2,1])
    
    k = sqrt(2*(β-α)/((1-α)*(1+β)))
    ρ = Elliptic.F(asin(√((1-α)/2)),k^2)
    
    K = Elliptic.K(k^2)
    Kp = Elliptic.K(1-k^2)
    q = exp(-π*Kp/K)
    
    Θ(x) = θ₄(x*π/(2K),q)
    H(x) = θ₁(x*π/(2K),q)
    Cₙ(n) = n==0 ? 1.0 : √2*Θ(ρ)/(√(Θ(ρ*(2n-1)))*√(Θ(ρ*(2n+1))))
    
    xx = iM(bands[1,1],bands[2,2])(x)
    if xx == α
        u = Kp*im
    else
        snu = √((α-1)*(1+xx)/(2*(α-xx)) |> Complex)#√(α-1 |> Complex)*√(1+x |> Complex)/√(2*(α-x) |> Complex)
        u = Elliptic.F(asin(snu),k^2)
    end
    int = -(1/(2im*π))*Cₙ(n)*((H(u-ρ)/H(u+ρ))^n*(Θ(u+2n*ρ)/Θ(u)))*√(xx-α |> Complex)/(√(xx-1 |> Complex)*√(xx+1 |> Complex)*√(xx-β |> Complex))
    int/((bands[2,2]-bands[1,1])/2)
end

function get_a(bands,nvec)
    α = iM(bands[1,1],bands[2,2])(bands[1,2])
    β = iM(bands[1,1],bands[2,2])(bands[2,1])
    
    k = sqrt(2*(β-α)/((1-α)*(1+β)))
    ρ = Elliptic.F(asin(√((1-α)/2)),k^2)
    
    K = Elliptic.K(k^2)
    Kp = Elliptic.K(1-k^2)
    q = exp(-π*Kp/K)
    
    Θ(x) = θ₄(x*π/(2K),q)
    dΘ(x) = dθ₄(x*π/(2K),q)*π/(2K)
    H(x) = θ₁(x*π/(2K),q)
    dH(x) = dθ₁(x*π/(2K),q)*π/(2K)
    
    ba = (1/(8Elliptic.Jacobi.sn(ρ,k^2)^2)+(d2sn(ρ,k^2)/(8Elliptic.Jacobi.sn(ρ,k^2)*(Elliptic.Jacobi.cn(ρ,k^2)*Elliptic.Jacobi.dn(ρ,k^2))^2))+(dH(2ρ)/H(2ρ))/(4Elliptic.Jacobi.sn(ρ,k^2)*Elliptic.Jacobi.cn(ρ,k^2)*Elliptic.Jacobi.dn(ρ,k^2)))*(1-α^2)
    
    coeffvec = zeros(length(nvec))
    for (i,n) in enumerate(nvec)
        if n == 0
            coeffvec[i] = (1-α^2)*(dΘ(ρ)/Θ(ρ))/(2Elliptic.Jacobi.sn(ρ,k^2)*Elliptic.Jacobi.cn(ρ,k^2)*Elliptic.Jacobi.dn(ρ,k^2))
        else
            coeffvec[i] = ((1-α^2)/(4Elliptic.Jacobi.sn(ρ,k^2)*Elliptic.Jacobi.cn(ρ,k^2)*Elliptic.Jacobi.dn(ρ,k^2)))*(dΘ((2n+1)*ρ)/Θ((2n+1)*ρ)-dΘ((2n-1)*ρ)/Θ((2n-1)*ρ))
        end
    end
    coeffvec.+=-ba+α
    coeffvec*(bands[2,2]-bands[1,1])/2 .+(bands[2,2]+bands[1,1])/2
end

function get_b(bands,nvec)
    α = iM(bands[1,1],bands[2,2])(bands[1,2])
    β = iM(bands[1,1],bands[2,2])(bands[2,1])
    
    k = sqrt(2*(β-α)/((1-α)*(1+β)))
    ρ = Elliptic.F(asin(√((1-α)/2)),k^2)
    
    K = Elliptic.K(k^2)
    Kp = Elliptic.K(1-k^2)
    q = exp(-π*Kp/K)
    
    Θ(x) = θ₄(x*π/(2K),q)
    dΘ(x) = dθ₄(x*π/(2K),q)*π/(2K)
    H(x) = θ₁(x*π/(2K),q)
    dH(x) = dθ₁(x*π/(2K),q)*π/(2K)
    
    coeffvec = zeros(length(nvec))
    for (i,n) in enumerate(nvec)
        if n==0
            coeffvec[i] = √(2Θ(3ρ)/Θ(ρ))*(dH(0)/H(2ρ))*(1-α^2)/(4Elliptic.Jacobi.sn(ρ,k^2)*Elliptic.Jacobi.cn(ρ,k^2)*Elliptic.Jacobi.dn(ρ,k^2))
        else
            coeffvec[i] = (1/(4Elliptic.Jacobi.sn(ρ,k^2)*Elliptic.Jacobi.cn(ρ,k^2)*Elliptic.Jacobi.dn(ρ,k^2)*H(2ρ)/(dH(0)*(1-α^2))))*√(Θ(ρ*(2n-1))*Θ(ρ*(2n+3)))/Θ((2n+1)*ρ)
        end
    end
    coeffvec*(bands[2,2]-bands[1,1])/2
end

function expg(bands, z)
    α = iM(bands[1,1],bands[2,2])(bands[1,2])
    β = iM(bands[1,1],bands[2,2])(bands[2,1])
    
    k = sqrt(2*(β-α)/((1-α)*(1+β)))
    ρ = Elliptic.F(asin(√((1-α)/2)),k^2)
    
    K = Elliptic.K(k^2)
    Kp = Elliptic.K(1-k^2)
    q = exp(-π*Kp/K)
    
    Θ(x) = θ₄(x*π/(2K),q)
    H(x) = θ₁(x*π/(2K),q)
    
    xx = iM(bands[1,1],bands[2,2])(z)
    if xx == α
        u = Kp*im
    else
        snu = √((α-1)*(1+xx)/(2*(α-xx)) |> Complex)#√(α-1 |> Complex)*√(1+x |> Complex)/√(2*(α-x) |> Complex)
        u = Elliptic.F(asin(snu),k^2)
    end
    #println(-H(Elliptic.F(asin(1e6im),k^2)+ρ)/H(Elliptic.F(asin(1e6im),k^2)-ρ))
    -H(u+ρ)/H(u-ρ)
end