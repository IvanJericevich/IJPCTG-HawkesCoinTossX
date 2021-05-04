#=
HawkesCalibration:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Tim Gebbie
- Function: Calibrate a 10-variate Hawkes process to CoinTossX and raw Hawkes data to obtain hypothesis tests
- Structure:
    1. CoinTossX Hawkes and Raw Hawkes calibration
    2. Retrieve data output
    3. Confidence intervals
    4. Hypothesis tests
    5. Distortion
    6. Impulse response
- Examples:
    ConfidenceIntervals(θ0, θ₀, RawHawkes, "Hawkes")
    ConfidenceIntervals(θ₁, θ₀, Model1data, "Model1")
    ConfidenceIntervals(θ₂, θ₀, Model2data, "Model2")
    Distortion(θ0, θ₀, "Hawkes")
    Distortion(θ₁, θ₀, "Model1")
    Distortion(θ₂, θ₀, "Model2")
    ImpulseResponse(θ₁, 10, Model1data, "Model1")
    ImpulseResponse(θ₂, 10, Model2data, "Model2")
=#
using DataFrames, Dates, Optim, CSV, ForwardDiff, Distributions, Plots, LaTeXStrings
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
@time calibratedParameters = optimize(logLikelihood, initialSolution, LBFGS(), Optim.Options(show_trace = true, iterations = 10000))
open("Data/Model1/Parameters.txt", "w") do file
    for p in exp.(Optim.minimizer(calibratedParameters))
        println(file, p)
    end
end

# Model 2
data = PrepareData("Model2/OrdersSubmitted_1", "Model2/Trades_1") |> x -> CleanData(x, allowCrossing = false) |> y -> PrepareHawkesData(y)
initialSolution = log.(vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1))))
logLikelihood = TwiceDifferentiable(θ -> Calibrate(exp.(θ), data, T, dimension), initialSolution, autodiff = :forward)
@time calibratedParameters = optimize(logLikelihood, initialSolution, LBFGS(), Optim.Options(show_trace = true, iterations = 10000))
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
calibratedParameters = optimize(logLikelihood, init, LBFGS(), Optim.Options(show_trace = true, iterations = 10000))
open("Data/Parameters.txt", "w") do file
    for p in exp.(Optim.minimizer(calibratedParameters))
        println(file, p)
    end
end

#---------------------------------------------------------------------------------------------------

#----- Data output -----#
# Read in data for model 1 and 2, also get data for raw hawkes
Model1data = PrepareData("Model1/OrdersSubmitted_1", "Model1/Trades_1") |> x -> CleanData(x, allowCrossing = true) |> y -> PrepareHawkesData(y)
Model2data = PrepareData("Model2/OrdersSubmitted_1", "Model2/Trades_1") |> x -> CleanData(x, allowCrossing = false) |> y -> PrepareHawkesData(y)
RawHawkes = ThinningSimulation(λ₀, α, β, T, seed = 1)

# Read in the parameters
θ₀ = vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1)))
θ0 = CSV.File("Data/Parameters.txt", header = false) |> Tables.matrix |> vec
RMSE0 = sqrt(mean((θ0 - θ₀).^2)); MAE0 = mean(abs.(θ0 - θ₀))
θ₁ = CSV.File("Data/Model1/Parameters.txt", header = false) |> Tables.matrix |> vec
RMSE₁ = sqrt(mean((θ₁ - θ₀).^2)); MAE0 = mean(abs.(θ₁ - θ₀))
θ₂ = CSV.File("Data/Model2/Parameters.txt", header = false) |> Tables.matrix |> vec
RMSE₂ = sqrt(mean((θ₂ - θ₀).^2)); MAE0 = mean(abs.(θ₂ - θ₀))
#---------------------------------------------------------------------------------------------------

#----- Confidence intervals -----#
function ConfidenceIntervals(θ::Vector{Float64}, θ₀::Vector{Float64}, data::Vector{Vector{Float64}}, model::String; format::String = "pdf")
    FIMθ = -ForwardDiff.hessian(params -> -Calibrate(params, data, 28800, 10), θ)
    Varθ = inv(FIMθ) ./ 28800
    σ = map(i -> Varθ[i, i] <= 0 ? NaN : sqrt(diag(Varθ)[i]), 1:210)
    # Standard parameter confidence intervals
    θᵀ = copy(θ)
    θᵀ[findall(x -> x > 10, θ)] .= NaN
    θᵀ = vcat(θᵀ[1:10], vec(transpose(reshape(θᵀ[(10 + 1):(10 * 10 + 10)], 10, 10))), vec(transpose(reshape(θᵀ[(end - 10 * 10 + 1):end], 10, 10))))
    σᵀ = vcat(σ[1:10], vec(transpose(reshape(σ[(10 + 1):(10 * 10 + 10)], 10, 10))), vec(transpose(reshape(σ[(end - 10 * 10 + 1):end], 10, 10))))
    θ₀ᵀ = vcat(θ₀[1:10], vec(transpose(reshape(θ₀[(10 + 1):(10 * 10 + 10)], 10, 10))), vec(transpose(reshape(θ₀[(end - 10 * 10 + 1):end], 10, 10))))
    confidenceIntervals = plot(1:210, θ₀ᵀ, seriestype = :scatter, marker = (:hline, :blue), label = "True", legend = :topleft, xlabel = "Indices", ylabel = "Estimates")
    plot!(confidenceIntervals, 1:210, θᵀ, seriestype = :scatter, yerror = 1.96 .* σᵀ, marker = (:hline, :black), label = "")
    savefig(confidenceIntervals, string("Figures/", model, "ConfidenceIntervals.", format))
    # Branching ratio confidence intervals
    α_hat = reshape(θ[(10 + 1):(10 * 10 + 10)], 10, 10)
    varα = reshape(diag(Varθ)[(10 + 1):(10 * 10 + 10)], 10, 10)
    β_hat = reshape(θ[(end - 10 * 10 + 1):end], 10, 10)
    varβ = reshape(diag(Varθ)[(end - 10 * 10 + 1):end], 10, 10)
    covαβ = Varθ[11:110, 111:end]
    σ² = zeros(10, 10)
    for i in 1:10
        for j in 1:10
            if varα[i, j] < 0 || varβ[i, j] < 0
                σ²[i, j] = NaN
            else
                ∂α = 1 / β_hat[i, j]
                ∂β = - α_hat[i, j] / (β_hat[i, j] ^ 2)
                σ²[i, j] = ∂α^2 * varα[i, j] + ∂β^2 * varβ[i, j] + 2 * ∂α * ∂β * covαβ[i, j]
                σ²[i, j] = σ²[i, j] < 0 ? NaN : σ²[i, j]
            end
        end
    end
    σ²ᵀ = vec(transpose(σ²))
    σ²ᵀ[findall(x -> x > 0.1, σ²ᵀ)] .= NaN
    confidenceIntervals = plot(1:100, vec(transpose(reshape(θ₀[(10 + 1):(10 * 10 + 10)] ./ θ₀[(end - 10 * 10 + 1):end], 10, 10))), seriestype = :scatter, marker = (:hline, :blue), label = "True", legend = :topleft, xlabel = "Indices", ylabel = "Estimates")
    plot!(confidenceIntervals, 1:100, vec(transpose(α_hat ./ β_hat)), seriestype = :scatter, yerror = 1.96 .* sqrt.(σ²ᵀ), marker = (:hline, :black), label = "")
    savefig(confidenceIntervals, string("Figures/", model, "BRConfidenceIntervals.", format))
end
#---------------------------------------------------------------------------------------------------

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
#---------------------------------------------------------------------------------------------------

#----- Distortion -----#
function Distortion(θ::Vector{Float64}, θ₀::Vector{Float64}, model::String; format = "pdf")
    # Alpha
    α = reshape(θ₀[(10 + 1):(10 * 10 + 10)], 10, 10)
    α_hat = reshape(θ[(10 + 1):(10 * 10 + 10)], 10, 10)
    αDistortion = α_hat - α
    colorGradient = model != "Hawkes" ? cgrad([:white, :red], scale = :exp) : cgrad([:blue, :white, :red], [0, 0.3, 0.4, 1])
    alphaDistortion = heatmap(1:10, 1:10, αDistortion, xticks = 1:10, yticks = 1:10, yflip = true, c = colorGradient, colorbar_title = L"\Delta \alpha")
    annotate!([(j,i, text(round(αDistortion[i,j], digits = 5), 5, :black, :center)) for i in 1:10 for j in 1:10], linecolor = :white)
    savefig(alphaDistortion, string("Figures/", model, "AlphaDistortion.", format))
    # Beta
    β = reshape(θ₀[(end - 10 * 10 + 1):end], 10, 10)
    β_hat = reshape(θ[(end - 10 * 10 + 1):end], 10, 10)
    βDistortion = β_hat - β
    colorGradient = model != "Hawkes" ? cgrad([:white, :red], scale = :exp) : cgrad([:blue, :white, :red], [0, 0.05, 0.1, 1])
    betaDistortion = heatmap(1:10, 1:10, βDistortion, xticks = 1:10, yticks = 1:10, yflip = true, c = colorGradient, colorbar_title = L"\Delta \beta")
    annotate!([(j,i, text(round(βDistortion[i,j], digits = 5), 5, :black, :center)) for i in 1:10 for j in 1:10], linecolor = :white)
    savefig(betaDistortion, string("Figures/", model, "BetaDistortion.", format))
    # Mu
    muDistortion = plot(1:10, θ[1:10] - θ₀[1:10], xlabel = "Event Number", ylabel = L"\Delta \mu", seriestype = :bar, linecolor = :black, legend = false, xticks = 1:10, fillcolor = :blue, ylim = (-0.009, 0.008))
    savefig(muDistortion, string("Figures/", model, "MuDistortion.", format))
    # Branching ratio
    colorGradient = cgrad([:blue, :white, :red], [0, 0.3, 0.4, 1])
    branchingRatioDistortion = heatmap(1:10, 1:10, (α_hat ./ β_hat) - (α ./ β), xticks = 1:10, yticks = 1:10, yflip = true, c = colorGradient, colorbar_title = L"\Delta \frac{\alpha}{\beta}")
    annotate!([(j,i, text(round(((α_hat ./ β_hat) - (α ./ β))[i,j], digits = 5), 5, :black, :center)) for i in 1:10 for j in 1:10], linecolor = :white)
    savefig(branchingRatioDistortion, string("Figures/", model, "BranchingRatioDistortion.", format))
end
#---------------------------------------------------------------------------------------------------

#----- Impulse response -----#
function ImpulseResponse(θ::Vector{Float64}, index::Int64, data::Vector{Vector{Float64}}, model::String; format::String = "pdf")
    labels = ["Buy (MO)", "Sell (MO)", "Aggressive Bid (LO)", "Aggressive Ask (LO)", "Passive Bid (LO)", "Passive Ask (LO)", "Aggressive Cancel Bid", "Aggressive Cancel Sell", "Passive Cancel Bid", "Passive Cancel Ask"]
    colors = [:red, :firebrick, :blue, :deepskyblue, :green, :seagreen, :purple, :mediumpurple, :yellow, :black]
    β_hat = reshape(θ[(end - 10 * 10 + 1):end], 10, 10)
    α_hat = reshape(θ[(10 + 1):(10 * 10 + 10)], 10, 10)
    branchingRatio = α_hat ./ β_hat
    impulseResponse = plot([log(2) / β_hat[index, 1]], [branchingRatio[index, 1]], seriestype = :scatter, marker = (colors[1], stroke(colors[1]), (length(data[1]) * 100) / sum(length.(data))), label = labels[1], xlabel = "Half-life effect (seconds)", ylabel = "Expected number of precipitated events", legend = :topright)
    for i in 2:10
        plot!(impulseResponse, [log(2) / β_hat[index, i]], [branchingRatio[index, i]], seriestype = :scatter, marker = (colors[i], stroke(colors[i]), (length(data[i]) * 100) / sum(length.(data))), label = labels[i])
    end
    savefig(impulseResponse, string("Figures/", model, "ImpulseResponse", index, ".", format))
end
#---------------------------------------------------------------------------------------------------




#=
Varθ₁ = inv(FIMθ₁)./ T
FIMθ₂ = -ForwardDiff.hessian(θ -> -Calibrate(θ, Model2data, 28800, 10), θ₂)
Varθ₂ = inv(FIMθ₂) ./ T
FIMθ0 = -ForwardDiff.hessian(θ -> -Calibrate(θ, RawHawkes, 28800, 10), θ0)
Varθ0 = inv(FIMθ0) ./ T

CIθ₁ = zeros(210, 2); CIθ₂ = zeros(210, 2); CIθ0 = zeros(210, 2)
for i in 1:210
    ## Getting negative variances for model 1 and 2. Not sure what is causing it
    # CIθ₁[i,1] = θ₁[i] - 1.96 * sqrt(Varθ₁[i,i]); CIθ₁[i,2] = θ₁[i] + 1.96 * sqrt(Varθ₁[i,i])
    # CIθ₂[i,1] = θ₂[i] - 1.96 * sqrt(Varθ₂[i,i]); CIθ₂[i,2] = θ₂[i] + 1.96 * sqrt(Varθ₂[i,i])
    CIθ0[i,1] = θ0[i] - 1.96 * sqrt(Varθ0[i,i]/T); CIθ0[i,2] = θ0[i] + 1.96 * sqrt(Varθ0[i,i]/T)
end
=#
