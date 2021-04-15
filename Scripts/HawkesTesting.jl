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

loglikeHawkes(t, lambda0, alpha, beta, T)

function Calibrate(param)
    lambda0 = [param[1] param[1]]
    alpha = [0 param[2]; param[2] 0]
    beta = [0 param[3]; param[3] 0]
    return -loglikeHawkes(t, lambda0, alpha, beta, T)
end

res = optimize(Calibrate, [0.015; 0.023; 0.11])
par = Optim.minimizer(res)

init = log.([0.015; 0.023; 0.11])
logLikelihood = TwiceDifferentiable(θ -> Calibrate(exp.(θ)), init, autodiff = :forward)
calibratedParameters = optimize(logLikelihood, init, LBFGS(), Optim.Options(show_trace = true, iterations = 5000))
par = exp.(Optim.minimizer(calibratedParameters))

hatλ₀ = [par[1]; par[2]]
hatα  = [0 par[2]; par[2] 0]
hatβ  = [0 par[3]; par[3] 0]
GR = GeneralisedResiduals(t, hatλ₀, hatα, hatβ)

mean(GR[1]); var(GR[1])
mean(GR[2]); var(GR[2])

# 10-variate
function Calibrate(θ::Vector{Type}, history::Vector{Vector{Float64}}, T::Int64, dimension::Int64) where Type <: Real # Maximum likelihood estimation
    λ₀ = θ[1:dimension]
    α = reshape(θ[(dimension + 1):(dimension * dimension + dimension)], dimension, dimension)
    β = reshape(θ[(end - dimension * dimension + 1):end], dimension, dimension)
    return -loglikeHawkes(history, λ₀, α, β, T)
end
λ₀ = [0.01; 0.01; 0.02; 0.02; 0.02; 0.02; 0.015; 0.015; 0.015; 0.015]
α = reduce(hcat, fill(λ₀, 10))
β  = fill(0.2, 10, 10)
T = 3600*8     # large enough to get good estimates, but not too long that it'll run for too long
Random.seed!(1)
t = ThinningSimulation(λ₀, α, β, T, seed = 1)
init = log.(vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1))))
logLikelihood = TwiceDifferentiable(θ -> Calibrate(exp.(θ), t, T, 10), init, autodiff = :forward)
calibratedParameters = optimize(logLikelihood, init, LBFGS(), Optim.Options(show_trace = true, iterations = 10000))
par = exp.(Optim.minimizer(calibratedParameters))




@time loglikeHawkes(t, λ₀, α, β, T)
