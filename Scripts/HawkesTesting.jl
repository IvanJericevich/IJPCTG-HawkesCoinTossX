#=
Hawkes
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Goal: Quick script to ensure all Hawkes functions are working appropriately
=#

using Optim, Statistics
include("Hawkes.jl")

lambda0 = [0.015;0.015]
alpha = [0 0.023; 0.023 0]
beta = [0 0.11; 0.11 0]

T = 3600*30     # large enough to get good estimates, but not too long that it'll run for too long
t = ThinningSimulation(lambda0, alpha, beta, T)

LogLikelihood(t, lambda0, alpha, beta, T)


function Calibrate2(θ::Vector{Type}, history::Vector{Vector{Float64}}, T::Int64, dimension::Int64) where Type <: Real # Maximum likelihood estimation
    λ₀ = [θ[1]; θ[2]]
    α  = [0 θ[2]; θ[2] 0]
    β  = [0 θ[3]; θ[3] 0]
    return -LogLikelihood(history, λ₀, α, β, T)
    # return -loglikeHawkes(history, λ₀, α, β, T)
end

init = log.([0.015; 0.023; 0.11])
logLikelihood = TwiceDifferentiable(θ -> Calibrate2(exp.(θ), t, T, 2), init, autodiff = :forward)
calibratedParameters = optimize(logLikelihood, init, LBFGS(), Optim.Options(show_trace = true, iterations = 5000))
par = exp.(Optim.minimizer(calibratedParameters))

hatλ₀ = [par[1]; par[2]]
hatα  = [0 par[2]; par[2] 0]
hatβ  = [0 par[3]; par[3] 0]
GR = GeneralisedResiduals(t, hatλ₀, hatα, hatβ)

mean(GR[1]); var(GR[1])
mean(GR[2]); var(GR[2])
