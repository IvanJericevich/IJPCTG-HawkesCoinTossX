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
FIMθ₁ = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model1data, 28800, 10), θ₁)
Varθ₁ = inv(FIMθ₁)
FIMθ₂ = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model2data, 28800, 10), θ₂)
Varθ₂ = inv(FIMθ₂)
FIMθ0 = -ForwardDiff.hessian(θ -> -Calibrate(θ, RawHawkes, 28800, 10), θ0)
Varθ0 = inv(FIMθ0)

CIθ₁ = zeros(210, 2); CIθ₂ = zeros(210, 2); CIθ0 = zeros(210, 2)
for i in 1:210
    ## Getting negative variances for model 1 and 2. Not sure what is causing it
    # CIθ₁[i,1] = θ₁[i] - 1.96 * sqrt(Varθ₁[i,i]); CIθ₁[i,2] = θ₁[i] + 1.96 * sqrt(Varθ₁[i,i])
    # CIθ₂[i,1] = θ₂[i] - 1.96 * sqrt(Varθ₂[i,i]); CIθ₂[i,2] = θ₂[i] + 1.96 * sqrt(Varθ₂[i,i])
    CIθ0[i,1] = θ0[i] - 1.96 * sqrt(Varθ0[i,i]/T); CIθ0[i,2] = θ0[i] + 1.96 * sqrt(Varθ0[i,i]/T)
end

plot(1:210, θ₀, seriestype = :scatter, markershape = :hline)
plot!(1:210, θ0, seriestype = :scatter, yerror = 1.96 .* sqrt.(diag(Varθ0)./T), markershape = :hline, color = :black)


#----- Hypothesis tests -----#
# Likelihood ratio test
2*(Calibrate(θ₀, Model1data, 28800, 10) - Calibrate(θ₁, Model1data, 28800, 10))
2*(Calibrate(θ₀, Model2data, 28800, 10) - Calibrate(θ₂, Model2data, 28800, 10))
2*(Calibrate(θ₀, RawHawkes, 28800, 10) - Calibrate(θ0, RawHawkes, 28800, 10))

cdf( Chisq(210), 2*(Calibrate(θ₀, Model1data, 28800, 10) - Calibrate(θ₁, Model1data, 28800, 10)))
cdf( Chisq(210), 2*(Calibrate(θ₀, Model2data, 28800, 10) - Calibrate(θ₂, Model2data, 28800, 10)))
cdf( Chisq(210), 2*(Calibrate(θ₀, RawHawkes, 28800, 10) - Calibrate(θ0, RawHawkes, 28800, 10)))

findall(x -> x<0, diag(Varθ₁))
findall(x -> x<0, diag(Varθ₂))
θ₁[findall(x -> x<0, diag(Varθ₁))]
θ₂[findall(x -> x<0, diag(Varθ₂))]
