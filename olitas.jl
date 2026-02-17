using Plots
import SpecialFunctions: beta_inc
import FFTW: rfft, irfft, fft, ifft
import UnicodePlots: lineplot

@kwdef struct WDMInfo
    Nf::Int
    Nt::Int
    dt::Float64 # timeseries sampling period
    N::Int = Nt*Nf
    T::Int = dt*N
    ΔT::Float64 = Nf*dt # width of one pixel in time
    ΔF::Float64 = 1/(2dt*Nf) # height of one pixel in freq 
    ΔΩ::Float64 = 2π*ΔF
    A::Float64 = 0 # Meyer shape parameter. note can't be bigger than ΔΩ/2
    n::Int = 4 # Window rolloff parameter

end

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

function ϕ(ω, wdm::WDMInfo)
    return ϕ(ω, wdm.ΔΩ, wdm.A, wdm.n)
end

function print_wdm_info(wdm::WDMInfo)
    println("================ WDM Basis Info ================")
    println("Nt: $(wdm.Nt)")
    println("Nf: $(wdm.Nf)")
    println("N:  $(wdm.N)")
    println("dt: $(wdm.dt)")
    println("Pixel duration:  $(wdm.ΔT)")
    println("Pixel bandwidth: $(wdm.ΔF)")
    println("Meyer window shape: A/ΔΩ = $(wdm.A/wdm.ΔΩ), n = $(wdm.n)")
    ωs = range(-wdm.ΔΩ, wdm.ΔΩ, length=100)
    phis = [ϕ(ωi,wdm) for ωi in ωs]
    plt = lineplot(ωs ./ wdm.ΔΩ, phis, xlabel="ω/ΔΩ", ylabel="ϕ(ω)")
    println(plt)
    println("================================================")
end

function build_filter(wdm::WDMInfo; forward=true)
    ΔΩ_s = π / wdm.Nf  # sample-space ΔΩ
    omegas = 2π / wdm.N .* (0:(wdm.Nt÷2))
    phif = [ϕ(om, ΔΩ_s, wdm.A, wdm.n) for om in omegas]
    nrm = sqrt( (2sum(phif[2:end].^2) + phif[1]^2) * 2/wdm.N)
    phif ./= nrm
    if forward
        phif .*= 2.0/wdm.Nf
    end
    phif
end
function inverse_wavelet_transform_freq(w, wdm::WDMInfo)
    if size(w) != (wdm.Nt, wdm.Nf)
        error("Input wavelet array dimensions $(size(w)) don't match given WDMInfo. (Nt=$(wdm.Nt), Nf=$(wdm.Nf))")
    end
    phif = build_filter(wdm,forward=false)
    out = zeros(Complex{eltype(w)}, wdm.N÷2 + 1)
    Nthalf = wdm.Nt÷2
    for m in 1:(wdm.Nf+1)
        center = (m-1)*Nthalf  # 0-based frequency index of layer center
        # repack time layers into a complex array
        prefactors = zeros(Complex{eltype(w)}, wdm.Nt)
        for n in 1:wdm.Nt
            if m==1
                prefactors[n] = w[(2*(n-1))%wdm.Nt+1, 1]/sqrt(2)
            elseif m==wdm.Nf+1
                prefactors[n] = w[(2*(n-1))%wdm.Nt+2, 1]/sqrt(2)
            else
                prefactors[n] = (n+m)%2 == 0 ? w[n,m] : -im*w[n,m]
            end
        end
        fft_result = fft(prefactors)
        imin = max(center - Nthalf + 1, 1)
        imax = min(center + Nthalf, wdm.N÷2 + 1)
        for i in imin:imax
            iind = abs((i-1) - center) + 1
            if iind > Nthalf + 1
                continue
            end
            if m==1 || m==(wdm.Nf+1)
                out[i] += fft_result[(2*(i-1)) % wdm.Nt + 1] * phif[iind]
            else
                out[i] += fft_result[(i-1) % wdm.Nt + 1] * phif[iind]
            end
        end
    end
    return out
end

function wavelet_transform_freq(fftdata, wdm::WDMInfo)
    w = zeros((wdm.Nt, wdm.Nf))
    phif = build_filter(wdm)
    Nthalf = wdm.Nt÷2
    for m in 1:(wdm.Nf+1)
        center = (m-1)*Nthalf # +1?
        #Cnm = (n+m)%2 ? im : 1
        # set up to make this time layer
        X = zeros(eltype(fftdata), wdm.Nt)
        for j in (-Nthalf+1):(Nthalf-1)
            freq_index = center + j + 1
            # no negative freqs of DC, nor anything above Nyquist
            if (m == 1 && freq_index < 1) || (m == wdm.Nf + 1 && freq_index > wdm.N÷2 + 1)
                continue
            end
            weight = phif[abs(j)+1]
            if j == 0 && (m == 1 || m == wdm.Nf + 1)
                # boundary layers have less dof
                weight /= 2
            end
            # TODO check index math
            X[Nthalf + j + 1] = weight * fftdata[freq_index]
        end
        xm = ifft(X)
        # extract wavelet coefficients
        # for the edge layers, there are only half the d.o.f.
        # make the same choice as Matt's code and pack DC and Nyquist into first layer
        if m == 1
            w[1:2:end,1] .= real.(xm[1:2:end]) .* sqrt(2.0)
        elseif m == wdm.Nf+1
            w[2:2:end,1] .= real.(xm[1:2:end]) .* sqrt(2.0)
        else
            # general layer
            for n in 1:wdm.Nt
                if (n+m)%2 == 0
                    w[n,m] = real(xm[n])
                else
                    s = isodd(m) ? 1 : -1
                    w[n,m] = s*imag(xm[n])
                end
            end
        end
    end
    return w
end

function wavelet_transform_timefreq(timeseries, wdm::WDMInfo)
    # this is not the actual time transform, we convert to freq first
    if length(timeseries) != wdm.N
        error("attempted WDM transform with data of length $(length(timeseries)), while WDMInfo is setup for N=$(wdm.N)")
    end
    X = rfft(timeseries)
    return wavelet_transform_freq(X, wdm)
end

using DelimitedFiles
function demo()
    wdm = WDMInfo(
                  Nf = 1536,
                  Nt = 10,
                  dt = 5.0
                 )
    print_wdm_info(wdm)

    testdata = zeros(wdm.N)
    testdata[1] = 1.0
    X = rfft(testdata)

    w = wavelet_transform_freq(X, wdm)
    writedlm("../ldasoft/py_wdm_impulse.dat", w)
    Xp = inverse_wavelet_transform_freq(w, wdm)
    timeseries_out = irfft(Xp, wdm.N)

    println("average roundtrip error per element (freq): $(sum(abs.(real.(X) - real.(Xp)) .+ abs.(imag.(X) - imag.(Xp)))/length(X))")
    println("average roundtrip error per element (time): $(sum(abs.(testdata-timeseries_out))/length(testdata))")
end
