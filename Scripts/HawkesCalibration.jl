#=
HawkesCalibration:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Calibrate a 10-variate Hawkes process to CoinTossX and raw Hawkes data to obtain hypothesis tests
- Structure:
    1. CoinTossX Hawkes and Raw Hawkes calibration
    2. Confidence intervals
    3. Hypothesis tests
=#
using DataFrames, Dates, Optim, CSV, ForwardDiff, Distributions
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
# Model 1
data = PrepareData("Model1/OrdersSubmitted_1", "Model1/Trades_1") |> x -> CleanData(x, allowCrossing = true) |> y -> PrepareHawkesData(y)
initialSolution = log.(vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1))))
logLikelihood = TwiceDifferentiable(θ -> Calibrate(exp.(θ), data, T, dimension), initialSolution, autodiff = :forward)
@time calibratedParameters = optimize(logLikelihood, initialSolution, LBFGS(), Optim.Options(show_trace = true, iterations = 2000))
open("Data/Model1/Parameters.txt", "w") do file
    for p in exp.(Optim.minimizer(calibratedParameters))
        println(file, p)
    end
end

# Model 2
data = PrepareData("Model2/OrdersSubmitted_1", "Model2/Trades_1") |> x -> CleanData(x, allowCrossing = false) |> y -> PrepareHawkesData(y)
initialSolution = log.(vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1))))
logLikelihood = TwiceDifferentiable(θ -> Calibrate(exp.(θ), data, T, dimension), initialSolution, autodiff = :forward)
@time calibratedParameters = optimize(logLikelihood, initialSolution, LBFGS(), Optim.Options(show_trace = true, iterations = 2000))
open("Data/Model2/Parameters.txt", "w") do file
    for p in exp.(Optim.minimizer(calibratedParameters))
        println(file, p)
    end
end

# Raw Hawkes
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

#=
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
=#

#---------------------------------------------------------------------------------------------------
# Read in data for model 1 and 2, also get data for raw hawkes
Model1data = PrepareData("Model1/OrdersSubmitted_1", "Model1/Trades_1") |> x -> CleanData(x, allowCrossing = true) |> y -> PrepareHawkesData(y)
Model2data = PrepareData("Model2/OrdersSubmitted_1", "Model2/Trades_1") |> x -> CleanData(x, allowCrossing = false) |> y -> PrepareHawkesData(y)
RawHawkes = ThinningSimulation(λ₀, α, β, T, seed = 1)

# Read in the parameters
θ₀ = vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1)))
θ0 = CSV.File("Data/Parameters.txt", header = false) |> Tables.matrix |> vec
θ₁ = CSV.File("Data/Model1/Parameters.txt", header = false) |> Tables.matrix |> vec
θ₂ = CSV.File("Data/Model2/Parameters.txt", header = false) |> Tables.matrix |> vec

#----- Confidence intervals -----#
Varθ₁ = inv(-ForwardDiff.hessian(θ -> -Calibrate(θ, Model1data, 28800, 10), θ₁))
Varθ₂ = inv(-ForwardDiff.hessian(θ -> -Calibrate(θ, Model2data, 28800, 10), θ₂))
Varθ0 = inv(-ForwardDiff.hessian(θ -> -Calibrate(θ, RawHawkes, 28800, 10), θ0))

CIθ₁ = zeros(210, 2); CIθ₂ = zeros(210, 2); CIθ0 = zeros(210, 2)
for i in 1:210
    CIθ₁[i,1] = θ₁[i] - 1.96 * sqrt(abs(Varθ₁[i,i])); CIθ₁[i,2] = θ₁[i] + 1.96 * sqrt(abs(Varθ₁[i,i]))
    CIθ₂[i,1] = θ₂[i] - 1.96 * sqrt(abs(Varθ₂[i,i])); CIθ₂[i,2] = θ₂[i] + 1.96 * sqrt(abs(Varθ₂[i,i]))
    CIθ0[i,1] = θ0[i] - 1.96 * sqrt(abs(Varθ0[i,i])); CIθ0[i,2] = θ0[i] + 1.96 * sqrt(abs(Varθ0[i,i]))
end

#----- Hypothesis tests -----#
# Likelihood ratio test
2*(Calibrate(θ₀, Model1data, 28800, 10) - Calibrate(θ₁, Model1data, 28800, 10))
2*(Calibrate(θ₀, Model2data, 28800, 10) - Calibrate(θ₂, Model2data, 28800, 10))
2*(Calibrate(θ₀, RawHawkes, 28800, 10) - Calibrate(θ0, RawHawkes, 28800, 10))

cdf( Chisq(210), 2*(Calibrate(θ₀, Model1data, 28800, 10) - Calibrate(θ₁, Model1data, 28800, 10)))
cdf( Chisq(210), 2*(Calibrate(θ₀, Model2data, 28800, 10) - Calibrate(θ₂, Model2data, 28800, 10)))
cdf( Chisq(210), 2*(Calibrate(θ₀, RawHawkes, 28800, 10) - Calibrate(θ0, RawHawkes, 28800, 10)))

#=
# Score test
Model1Score = ForwardDiff.gradient(θ -> -Calibrate(θ, Model1data, 28800, 10), θ₀)
Model1Fisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model1data, 28800, 10), θ₁)
Model1ScoreTest = Model1Score' * inv(Model1Fisher) * Model1Score

Model2Score = ForwardDiff.gradient(θ -> -Calibrate(θ, Model2data, 28800, 10), θ₀)
Model2Fisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model2data, 28800, 10), θ₂)
Model2ScoreTest = Model2Score' * inv(Model2Fisher) * Model2Score

RawScore = ForwardDiff.gradient(θ -> -Calibrate(θ, RawHawkes, 28800, 10), θ₀)
RawFisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, RawHawkes, 28800, 10), θ0)
RawScoreTest = RawScore' * inv(RawFisher) * RawScore

# Wald test
Model1Fisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model1data, 28800, 10), θ₁)
(θ₁ .- θ₀)' * Model1Fisher * (θ₁ .- θ₀)

Model2Fisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model2data, 28800, 10), θ₂)
(θ₂ .- θ₀)' * Model2Fisher * (θ₂ .- θ₀)

RawFisher = -ForwardDiff.hessian(θ -> -Calibrate(θ, RawHawkes, 28800, 10), θ0)
(θ0 .- θ₀)' * RawFisher * (θ0 .- θ₀)
=#
