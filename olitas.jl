using Plots
import SpecialFunctions: beta_inc
import FFTW: rfft, rfftfreq

function ν(x, n::Int = 4)
    # beta_inc(a,b,x,y=1-x)
    return beta_inc(n,n,x)[1] / beta_inc(n,n,1)[1]
end

function ϕ(ω, ΔΩ, A=0, n::Int = 4)
    # This is the generalized Meyer window function
    B = ΔΩ - 2A
    if abs(ω) < A
        return 1.0/sqrt(ΔΩ)
    elseif abs(ω) < (A + B)
        return 1.0/sqrt(ΔΩ) * cos(π/2 * ν((abs(ω) - A)/B, n))
    else
        return zero(ω)
    end
end
# following notation from Neil, 2009.00043
function g(ω, m, n, wdm::WDMInfo)
    Cnm = (n+m)%2 ? im : 1
    return exp(-im*n*ω*wdm.ΔT)*(Cnm*ϕ(ω-m*wdm.ΔΩ, wdm) + conj(Cnm)*ϕ(ω+m*wdm.ΔΩ, wdm))
end

@kwdef struct WDMInfo
    Nf::Int
    Nt::Int
    dt::Float64 # timeseries sampling period
    #time_pixel_duration::Float64 = 7680.0
    #bandwidth::Float64 = 0.0
    N::Int = Nt*Nf
    T::Int = dt*N
    ΔT::Float64 = Nf*dt # width of one pixel in time
    ΔF::Float64 = 1/(2dt*Nf) # height of one pixel in freq 
    ΔΩ::Float64 = 2π*ΔF
    A::Float64 = 0 # Meyer shape parameter. note can't be bigger than ΔΩ/2
    n::Int = 4 # Window rolloff parameter

end
function ϕ(ω, wdm::WDMInfo)
    return ϕ(ω, wdm.ΔΩ, wdm.A, wdm.n)
end

function print_wdm_info(wdm::WDMInfo)
    println("======== WDM Basis Info ========")
    println("Nt: $(wdm.Nt)")
    println("Nf: $(wdm.Nf)")
    println("N:  $(wdm.N)")
    println("dt: $(wdm.dt)")
    println("Pixel duration:  $(wdm.ΔT)")
    println("Pixel bandwidth: $(wdm.ΔF)")
    println("Meyer window shape: A/ΔΩ=$(wdm.A/wdm.ΔΩ), n = $(wdm.n)")
    println("================================")
end

function wavelet_transform(timeseries, wdm::WDMInfo)
    # this is not the actual time transform, we convert to freq first
    if length(timeseries) != wdm.N
        error("attempted WDM transform with data of length $(length(timeseries)), while WDMInfo is setup for N=$(wdm.N)")
    end
    X = rfft(timeseries)
    return X
end

function demo()
    wdm = WDMInfo(
                  Nf = 1536,
                  Nt = 7,
                  #Nf = 100,
                  #Nt = 100,
                  dt = 5.0
                 )
    print_wdm_info(wdm)
    #=
    ωs = range(-5wdm.ΔΩ,5wdm.ΔΩ,length=100)
    phis = [ϕ(ωi,wdm) for ωi in ωs]
    plot(ωs, phis)
    gui()
    =#

    testdata = zeros(wdm.N)
    testdata[1] = 1.0
    wavelet_transform(testdata, wdm)

end
