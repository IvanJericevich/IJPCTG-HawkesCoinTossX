#=
HawkesCalibration:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Calibrate a 10-variate Hawkes process to CoinTossX and raw Hawkes data to obtain hypothesis tests
- Structure:
    1. CoinTossX Hawkes calibration
    2. Raw Hawkes calibration
    3. Hypothesis tests and confidence intervals
=#
using DataFrames, Dates, Optim, CSV, ForwardDiff
clearconsole()
include(pwd() * "/Scripts/Hawkes.jl")
include(pwd() * "/Scripts/DataCleaning.jl")
λ₀ = [0.01; 0.01; 0.02; 0.02; 0.02; 0.02; 0.015; 0.015; 0.015; 0.015]
α = reduce(hcat, fill(λ₀, 10))
β  = fill(0.2, 10, 10)
T = 3600*8
dimension = 10
#---------------------------------------------------------------------------------------------------


#----- CoinTossX Hawkes calibration -----#
data = PrepareData("Model2/OrdersSubmitted_1", "Model2/Trades_1") |> x -> CleanData(x, allowCrossing = false) |> y -> PrepareHawkesData(y)
initialSolution = log.(vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1))))
logLikelihood = TwiceDifferentiable(θ -> Calibrate(exp.(θ), data, T, dimension), initialSolution, autodiff = :forward)
@time calibratedParameters = optimize(logLikelihood, initialSolution, LBFGS(), Optim.Options(show_trace = true, iterations = 2000))
open("Parameters2.txt", "w") do file
    for p in exp.(Optim.minimizer(calibratedParameters))
        println(file, p)
    end
end
θ₁ = CSV.File("Data/Model1/Parameters.txt", header = false) |> Tables.matrix |> vec
MAE₁ = mean(abs.(θ₁ - exp.(initialSolution)))
RMSE₁ = sqrt(mean((θ₁ - exp.(initialSolution)) .^ 2))
θ₂ = CSV.File("Data/Model2/Parameters.txt", header = false) |> Tables.matrix |> vec
MAE₂ = mean(abs.(θ₂ - exp.(initialSolution)))
RMSE₂ = sqrt(mean((θ₂ - exp.(initialSolution)) .^ 2))
distortion1 = (θ₁ - exp.(initialSolution)) ./ exp.(initialSolution)
distortion2 = (θ₂ - exp.(initialSolution)) ./ exp.(initialSolution)
α = reshape(θ₁[(dimension + 1):(dimension * dimension + dimension)], dimension, dimension)
β = reshape(θ₁[(end - dimension * dimension + 1):end], dimension, dimension)
α ./ β # Branching ratio
#---------------------------------------------------------------------------------------------------

#----- Raw Hawkes calibration -----#
Random.seed!(1)
t = ThinningSimulation(λ₀, α, β, T, seed = 1)
init = log.(vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1))))
logLikelihood = TwiceDifferentiable(θ -> Calibrate(exp.(θ), t, T, dimension), init, autodiff = :forward)
calibratedParameters = optimize(logLikelihood, init, LBFGS(), Optim.Options(show_trace = true, iterations = 2000))
open("Data/Parameters.txt", "w") do file
    for p in exp.(Optim.minimizer(calibratedParameters))
        println(file, p)
    end
end
θ = exp.(Optim.minimizer(calibratedParameters))
#---------------------------------------------------------------------------------------------------

#----- Hypothesis tests and confidence intervals -----#
# Fits the score and Fisher info with model data using original params to see if the true params still work.
θ₀ = vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1)))

Model1data = PrepareData("Model1/OrdersSubmitted_1", "Model1/Trades_1") |> x -> CleanData(x, allowCrossing = true) |> y -> PrepareHawkesData(y)
Model1Score = ForwardDiff.gradient(θ -> -Calibrate(θ, Model1data, 28800, 10), θ₀)
Model1Fisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model1data, 28800, 10), θ₀)
Model1ScoreTest = Model1Score' * inv(Model1Fisher) * Model1Score

Model2data = PrepareData("Model2/OrdersSubmitted_1", "Model2/Trades_1") |> x -> CleanData(x, allowCrossing = false) |> y -> PrepareHawkesData(y)
Model2Score = ForwardDiff.gradient(θ -> -Calibrate(θ, Model2data, 28800, 10), θ₀)
Model2Fisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model2data, 28800, 10), θ₀)
Model2ScoreTest = Model2Score' * inv(Model2Fisher) * Model2Score

RawScore = ForwardDiff.gradient(θ -> -Calibrate(θ, t, 28800, 10), θ₀)
RawFisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, t, 28800, 10), θ₀)
RawScoreTest = RawScore' * inv(RawFisher) * RawScore

H = ForwardDiff.hessian(θ -> -Calibrate(exp.(θ), data, 28800, 10), Optim.minimizer(calibratedParameters))
𝓘 = inv(-H)
lam = θ₀[1:dimension]
a = reshape(θ₀[(dimension + 1):(dimension * dimension + dimension)], dimension, dimension)
b = reshape(θ₀[(end - dimension * dimension + 1):end], dimension, dimension)

Calibrate(θ₀, Model1data, 28800, 10) / Calibrate(θ₁, Model1data, 28800, 10)
Calibrate(θ₀, Model2data, 28800, 10) / Calibrate(θ₂, Model2data, 28800, 10)

-2*log(Calibrate(θ₀, Model1data, 28800, 10) / Calibrate(θ₁, Model1data, 28800, 10))
-2*log(Calibrate(θ₀, Model2data, 28800, 10) / Calibrate(θ₂, Model2data, 28800, 10))

cdf( Chisq(1), -2*log(Calibrate(θ₀, Model2data, 28800, 10) / Calibrate(θ₂, Model2data, 28800, 10)))
