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
