#=
Hawkes
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Provide the necessary functions for calibrating and simulating a multivariate Hawkes process (Toke and Pomponio 2012)
- Structure:
	1. Supplementary functions
	2. Simulation by thinning
	3. Intensity path
	4. Recursive relation
	5. Integrated intensity
	6. Log-likelihood objective
	7. Hawkes moments
	8. Method of moments objective
	9. Calibration
	10. Generalised residuals
	11. Validation plots and statistics
- Symbols:
	α = DxD matrix of excitations
    β = DxD matrix of rates of decay
    λ₀ = D-vector of baseline intensities
    u = Random sample from U(0, 1)
    λₜ = D-vector of intensities for each process at time t
    ∂λ = DxD matrix of intensity jumps
    τ = Inter-arrival time sampled/obtained from the inverse transform method
    Γ = Spectral radius to check stability conditions of the process
    λ_star = The cummulative value of the intensity of all processes
- TODO: Implement GPU parallel processing
=#
using Random, LinearAlgebra#, LaTeXStrings
#---------------------------------------------------------------------------------------------------

#----- Supplementary functions -----#
# Returns the Spectral Radius of Γ = A / B to check if stability conditions have been met (to ensure stationarity) (Toke-Pomponio (2011) - Modelling Trades-Through in a Limited Order-Book)
function SpectralRadius(alpha, beta)
    dimension = size(alpha)[1]
    Γ = zeros(dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if beta[i,j] != 0
                Γ[i,j] = alpha[i,j] / beta[i,j]
            end
        end
    end
    eigenval = eigen(Γ).values
    eigenval = abs.(eigenval)
    return maximum(eigenval)
end
# Returns the index of the process to which a sampled event can be attributed
function attribute(D, I_star, m_lambda)
    index = 1
    cumulative = m_lambda[1]

    while D > (cumulative/I_star)
        index = index + 1
        cumulative += m_lambda[index]
    end
    return index
end
#---------------------------------------------------------------------------------------------------

#----- Simulation by thinning -----#
# Returns vectors of sampled times from the multivariate D-type Hawkes process
function ThinningSimulation(lambda0, alpha, beta, T; kwargs...)

    kwargs = Dict(kwargs)

    if haskey(kwargs, :seed)
        seed = kwargs[:seed]
    else
        seed = 1
    end

    # Initialize
    dimension = length(lambda0)
    history = Vector{Vector{Float64}}()
    for i in 1:dimension
        history = push!(history, [])
    end

    dlambda = zeros(dimension, dimension)
    m_lambda0 = lambda0
    m_lambda = zeros(dimension, 1)

    SR = SpectralRadius(alpha, beta)
    if SR >= 1
        return println("WARNING: Unstable, Spectral Radius of Γ = A/B must be less than 1")
    end

    lambda_star = 0.0
    t = 0.0

    for i in 1:dimension
        lambda_star += m_lambda0[i]
        m_lambda[i] = m_lambda0[i]
    end

    # First Event
    Random.seed!(seed)
    U = rand()
    s = - log(U)/lambda_star

    if s <= T
        D = rand()
        n0 = attribute(D, lambda_star, m_lambda)
        history[n0] = append!(history[n0], s)

        for i in 1:dimension
            dlambda[i,n0] = alpha[i,n0]
            m_lambda[i] = m_lambda0[i] + alpha[i,n0]
        end
    else
        return history
    end

    t = s

    # General Routine
    lambda_star = 0.0
    for i in 1:dimension
        lambda_star = lambda_star + m_lambda[i]
    end

    while true
        seed += 1
        Random.seed!(seed)

        U = rand()
        s = s - (log(U) / lambda_star)

        seed += 1
        Random.seed!(seed)
        if s <= T
            D = rand()
            I_M = 0.0
            for i in 1:dimension
                dl = 0.0
                for j in 1:dimension
                    dl += dlambda[i,j] * exp(-beta[i,j] * (s - t))
                end
                m_lambda[i] = m_lambda0[i] + dl
                I_M = I_M + m_lambda[i]
            end

            if D <= (I_M / lambda_star)
                n0 = attribute(D, lambda_star, m_lambda)
                history[n0] = append!(history[n0], s)
                lambda_star = 0.0
                for i in 1:dimension
                    dl = 0.0
                    for j in 1:dimension
                        dlambda[i,j] = dlambda[i,j] * exp(-beta[i,j] * (s - t))
                        if n0 == j
                            dlambda[i,n0] += alpha[i,n0]
                        end
                        dl += dlambda[i,j]
                    end
                    lambda_star += m_lambda0[i] + dl
                end
                t = s
            else
                lambda_star = I_M
            end
        else
            return history
        end
    end
end
#---------------------------------------------------------------------------------------------------

#----- Intensity path -----#
# Extract the Intensity fuction given the simulation paths
function Intensity(m::Int64, time::Vector{Type}, history::Vector{Vector{Type}}, λ₀::Vector{Float64}, α::Array{Float64, 2}, β::Array{Float64, 2}) where Type <: Real
    λ = fill(λ₀[m], length(time))
    dimension = length(λ₀)
    for t in 1:length(time)
        for j in 1:dimension
            for tʲₖ in history[j]
                if tʲₖ < time[t]
                    λ[t] += α[m, j] * exp(- β[m, j] * (time[t] - tʲₖ))
                end
            end
        end
    end
    return λ
end
#---------------------------------------------------------------------------------------------------

#----- Recursive relation -----#
# Supporting function to calculate the recursive function R^{ij}(l) in the loglikelihood for a multivariate Hawkes process (Toke-Pomponio (2011) - Modelling Trades-Through in a Limited Order-Book)
function recursion(history, beta, m, n)
    history_m = history[m]
    history_m = append!([0.0], history_m)
    N = length(history_m)
    R = zeros(Real, N, 1)
    history_n = history[n]
    beta = beta[m,n]
    ix = Int(1)
    for i in 2:N
        if n == m
            R[i] = exp(-beta * (history_m[i] - history_m[i-1])) * (1 + R[i-1])
        else
            R[i] = exp(-beta * (history_m[i] - history_m[i-1])) * R[i-1]
            for j in ix:length(history_n)
                if history_n[j] >= history_m[i-1]
                    if history_n[j] < history_m[i]
                        R[i] += exp(-beta*(history_m[i] - history_n[j]))
                    else
                        ix = j
                        break
                    end
                end
            end
        end
    end
    return R[2:end]
end
#---------------------------------------------------------------------------------------------------

#----- Integrated intensity -----#
# Function to compute the integrated intensity from [0,T] ∫_0^T λ^m(t) dt in the loglikelihood for a multivariate Hawkes process
function Λ_m(history, T, lambda0, alpha, beta, m)
    Λ = lambda0[m] * T
    dimension = length(lambda0)

    Γ = zeros(Real, dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if beta[i,j] != 0
                Γ[i,j] = Real(alpha[i,j] / beta[i,j])
            end
        end
    end

    for n in 1:dimension
        for i in 1:length(history[n])
            if history[n][i] <= T
                Λ += Γ[m,n] * (1 - exp(-beta[m,n] * (T - history[n][i])))
            end
        end
    end
    return Λ
end
#---------------------------------------------------------------------------------------------------

#----- Log-likelihood objective -----#
# Computes the partial log-likelihoods and sums them up to obtain the full log-likelihood
function loglikeHawkes(history, lambda0, alpha, beta, T)
    dimension = length(lambda0)
    ll = zeros(Real, dimension, 1)

    for m in 1:dimension
        ll[m] = T - Λ_m(history, T, lambda0, alpha, beta, m)
        R_mn = zeros(Real, length(history[m]), dimension)
        for n in 1:dimension
            R_mn[:,n] = recursion(history, beta, m, n)
        end

        ind = findall(x-> x.< 0, R_mn)
        R_mn[ind] .= 0

        for l in 1:length(history[m])
            d = lambda0[m] + sum(alpha[m,:] .* R_mn[l,:])
            if d > 0
                ll[m] += log(d)
            else
                ll[m] += -100
            end
        end
    end
    return sum(ll)
end
#---------------------------------------------------------------------------------------------------

#----- Calibration -----#
# Functions to be used in the optimization routine (the below objectives should be minimized)
#=
function Calibrate(θ::Vector{Type}, history::Vector{Vector{Float64}}, T::Int64, dimension::Int64) where Type <: Real # Maximum likelihood estimation
    λ₀ = θ[1:dimension]
    α = reshape(θ[(dimension + 1):(dimension * dimension + dimension)], dimension, dimension)
    β = reshape(θ[(end - dimension * dimension + 1):end], dimension, dimension)
    return -LogLikelihood(history, λ₀, α, β, T)
end
=#
#---------------------------------------------------------------------------------------------------

#----- Generalised residuals -----#
function GeneralisedResiduals(history, lambda0, alpha, beta)
    # Initialize
    dimension = length(lambda0)
    GE = Vector{Vector{Float64}}()
    for i in 1:dimension
        GE = push!(GE, [])
    end
    # Loop through each dimension
    for m in 1:dimension
        # Initialise the integrated intensity
        Λ = zeros(length(history[m]), 1)
        # Loop through the observations in each process
        for l in 1:length(history[m])
            Λ[l] = Λ_m(history, history[m][l], lambda0, alpha, beta, m)
        end
        # Compute the error and push it into Generalised Errors
        for l in 2:length(history[m])
            # Append results
            GE[m] = append!(GE[m], Λ[l] - Λ[l-1])
        end
    end
    return GE
end
#---------------------------------------------------------------------------------------------------

#----- Validation plots and statistics -----#
#=
function Validate(simulation::Vector{Vector{Type}}, λ₀::Vector{Float64}, α::Array{Float64, 2}, β::Array{Float64, 2}, T::Int64; format::String = "pdf") where Type <: Real
	titles = ["BuyMO", "SellMO", "AggressiveBuyLO", "AggressiveSellLO", "PassiveBuyLO", "PassiveSellLO", "AggressiveBuyOC", "AggressiveSellOC", "PassiveBuyOC", "PassiveSellOC"]
	colors = [:red, :firebrick, :blue, :deepskyblue, :green, :seagreen, :purple, :mediumpurple, :yellow, :black]
	dimension = length(λ₀)
	for m in 1:dimension
		integratedIntensities = Λ(simulation, T, λ₀, α, β, m)
		# QQ plots
	    qqPlot = qqplot(Exponential(1), integratedIntensities, xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", title = titles[m], marker = (3, colors[m], stroke(colors[m])), linecolor = :black, legend = false)
	    savefig(qqPlot, string("Figures/QQPlot", titles[m], ".", format))
		# Independence plots
		Uᵢ = cdf.(Exponential(1), integratedIntensities)[1:(end - 1)]
        Uᵢ₊₁ = cdf.(Exponential(1), integratedIntensities)[2:end]
        independencePlot = plot(Uᵢ, Uᵢ₊₁, seriestype = :scatter, marker = (3, colors[m], colors[m]), title = titles[m], xlabel = L"U_k = F_{Exp(1)}(t_k - t_{k - 1})", ylabel = L"U_{k + 1} = F_{Exp(1)}(t_{k + 1} - t_k)", legend = false)
        savefig(independencePlot, string("Figures/IndependencePlot", titles[m], ".", format))
		# Statistical tests
        LBTest = LjungBoxTest(integratedIntensities, 20, 3) # Ljung-Box - H_0 = independent
		KSTest = ExactOneSampleKSTest(integratedIntensities, Exponential(1)) # KS - H_0 = exponential
        println(string(titles[m], "|", "LjungBox:Q=", round(LBTest.Q, digits = 5), ",p=", round(pvalue(LBTest), digits = 5), "|", "KolmogorovSmirnov:δ=", round(KSTest.δ, digits = 4), ",p=", round(pvalue(KSTest), digits = 4)))
	end
end
=#
