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
function SpectralRadius(α::Array{Float64, 2}, β::Array{Float64, 2})
    dimension = size(α, 1)
    Γ = zeros(dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if β[i,j] != 0
                Γ[i,j] = α[i,j] / β[i,j]
            end
        end
    end
    eigenvalue = abs.(eigen(Γ).values)
    if maximum(eigenvalue) >= 1
        error("Unstable, Spectral Radius of Γ = A/B must be less than 1")
    end
end
# Returns the index of the process to which a sampled event can be attributed
function Attribute(u::Float64, λ_star::Float64, λₜ::Vector{Float64})
    index = 1
    cummulativeIntensity = λₜ[1]
    while u > (cummulativeIntensity / λ_star)
        index += 1
        cummulativeIntensity += λₜ[index]
    end
    return index
end
#---------------------------------------------------------------------------------------------------

#----- Simulation by thinning -----#
# Returns vectors of sampled times from the multivariate D-type Hawkes process
function ThinningSimulation(λ₀::Vector{Float64}, α::Array{Float64, 2}, β::Array{Float64, 2}, T::Int64; seed::Int64 = 1)
    Random.seed!(seed)
    SpectralRadius(α, β)
    # Initialization
    dimension = length(λ₀)
    history = [Vector{Float64}() for _ in 1:dimension]
    ∂λ = zeros(dimension, dimension)
    λₜ = λ₀
    λ_star = sum(λ₀)
    t = 0.0
    # First event
    u = rand()
    s = - log(u) / λ_star # The first arrival is simply the inter arrival time to next event
    if s <= T
        u = rand()
        m = Attribute(u, λ_star, λₜ)
        push!(history[m], s)
        λₜ = λ₀ .+ α[:, m]
        ∂λ[:, m] = α[:, m]
    else
        error("First event time is greater than the horizon")
    end
    # General Routine
    t = s
    λ_star = sum(λₜ)
    Juno.progress() do id # Progress bar
        while true
            u = rand()
            τ = - log(u) / λ_star # Sample inter-arrival time using inverse-transform method
            s += τ
            if s <= T
                u = rand()
                λₜ = λ₀ .+ vec(sum(∂λ .* exp.(-β .* (s - t)), dims = 2)) # Sum across columns (index j) for each row
                if u <= (sum(λₜ) / λ_star) # Apply acceptance-rejection
                    m = Attribute(u, λ_star, λₜ)
                    push!(history[m], s)
                    λ_star = 0.0
                    ∂λ = ∂λ .* exp.(-β .* (s - t))
                    ∂λ[:, m] += α[:, m]
                    λ_star = sum(λ₀ .+ ∂λ)
                    t = s
                else
                    λ_star = sum(λₜ)
                end
            else
                return history
            end
            @info "Simulating" progress=(s / T) _id=id # Update progress
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
function R(history::Vector{Vector{Float64}}, β::Array{Type, 2}, i::Int64, j::Int64) where Type <: Real
    tⁱ = vcat([0.0], history[i]); tʲ = history[j]
    N = length(tⁱ)
    Rⁱᴶ = zeros(Type, N)
	ix = 1
    for n in 2:N
        if i == j
            Rⁱᴶ[n] = exp(- β[i, j] * (tⁱ[n] - tⁱ[n - 1])) * (1 + Rⁱᴶ[n - 1])
        else
			Rⁱᴶ[n] = exp(- β[i, j] * (tⁱ[n] - tⁱ[n - 1])) * Rⁱᴶ[n - 1]
            for m in ix:length(tʲ)
                if tʲ[m] >= tⁱ[n - 1]
                    if tʲ[m] < tⁱ[n]
                        Rⁱᴶ[n] += exp(- β[i, j] * (tⁱ[n] - tʲ[m]))
                    else
                        ix = m
                        break
                    end
                end
            end
        end
    end
    return Rⁱᴶ[2:end]
end
#---------------------------------------------------------------------------------------------------

#----- Integrated intensity -----#
# Function to compute the integrated intensity from [0,T] ∫_0^T λ^m(t) dt in the loglikelihood for a multivariate Hawkes process
function Λ(history::Vector{Vector{Float64}}, T::Int64, λ₀::Vector{Type}, α::Array{Type, 2}, β::Array{Type, 2}, m::Int64) where Type <: Real
    Λ = λ₀[m] * T
    dimension = length(λ₀)
    for j in 1:dimension
        if β[m, j] != 0
            for tₙ in history[j]
				if tₙ <= T
                	Λ += (α[m, j] / β[m, j]) * (1 - exp(-β[m, j] * (T - tₙ)))
				end
            end
        end
    end
    return Λ
end
#---------------------------------------------------------------------------------------------------

#----- Log-likelihood objective -----#
# Computes the partial log-likelihoods and sums them up to obtain the full log-likelihood
function LogLikelihood(history::Vector{Vector{Float64}}, λ₀::Vector{Type}, α::Array{Type, 2}, β::Array{Type, 2}, T::Int64) where Type <: Real
    dimension = length(λ₀)
    loglikelihood = Vector{Type}(undef, dimension)
    for m in 1:dimension
        loglikelihood[m] = T - Λ(history, T, λ₀, α, β, m)
        Rⁱᴶ = zeros(Type, length(history[m]), dimension)
        for j in 1:dimension
            Rⁱᴶ[:, j] = R(history, β, m, j)
        end
        loglikelihood[m] += sum(map(l -> log(λ₀[m] + sum(α[m, :] .* Rⁱᴶ[l, :])), 1:length(history[m])))
    end
    return sum(loglikelihood)
end
#---------------------------------------------------------------------------------------------------

#----- Calibration -----#
# Functions to be used in the optimization routine (the below objectives should be minimized)
function Calibrate(θ::Vector{Type}, history::Vector{Vector{Float64}}, T::Int64, dimension::Int64) where Type <: Real # Maximum likelihood estimation
    λ₀ = θ[1:dimension]
    α = reshape(θ[(dimension + 1):(dimension * dimension + dimension)], dimension, dimension)
    β = reshape(θ[(end - dimension * dimension + 1):end], dimension, dimension)
    return -LogLikelihood(history, λ₀, α, β, T)
end
#---------------------------------------------------------------------------------------------------

#----- Generalised residuals -----#
function GeneralisedResiduals(history::Vector{Vector{Type}}, λ₀::Vector{Float64}, α::Array{Float64, 2}, β::Array{Float64, 2}) where Type <: Real
    dimension = length(λ₀)
    GE = [Vector{Float64}() for _ in 1:dimension]
    for m in 1:dimension # Loop through each dimension
		integratedIntensity = map(t -> Λ(history, t, λ₀, α, β, m), history[m]) # Loop through the observations in each process
		GE[m] = diff(integratedIntensity) # Compute the error
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
