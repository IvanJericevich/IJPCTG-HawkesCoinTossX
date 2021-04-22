#=
HawkesCalibration:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Calibrate a 10-variate Hawkes process to CoinTossX and raw Hawkes data to obtain hypothesis tests
- Structure:
    1.
- Examples
	Validation(1, format = "png")
=#
using HypothesisTests, Distributions
clearconsole()
include(pwd() * "/Scripts/Hawkes.jl")
include(pwd() * "/Scripts/DataCleaning.jl")
#---------------------------------------------------------------------------------------------------

#----- Validation plots and statistics -----#
function Validation(model::Int64; dimension::Int64 = 10, format = "pdf")
	θ = CSV.File(string("Data/Model", model, "/Parameters.txt"), header = false) |> Tables.matrix |> vec # Calibrated parameters
	λ₀ = θ[1:dimension]
	α = reshape(θ[(dimension + 1):(dimension * dimension + dimension)], dimension, dimension)
	β = reshape(θ[(end - dimension * dimension + 1):end], dimension, dimension)
	allowCrossing = model == 1 ? true : false
	data = PrepareData(string("Model", model, "/OrdersSubmitted_1"), string("Model", model, "/Trades_1")) |> x -> CleanData(x, allowCrossing = allowCrossing) |> y -> PrepareHawkesData(y) # Simulated data
	integratedIntensities = GeneralisedResiduals(data, λ₀, α, β)
	LBTest = fill(0.0, 10, 2); KSTest = fill(0.0, 10, 2) # Initialize test statistics
	colors = [:red, :firebrick, :blue, :deepskyblue, :green, :seagreen, :purple, :mediumpurple, :yellow, :black]
	labels = ["Buy (MO)", "Sell (MO)", "Aggressive Bid (LO)", "Aggressive Ask (LO)", "Passive Bid (LO)", "Passive Ask (LO)", "Aggressive Cancel Bid", "Aggressive Cancel Sell", "Passive Cancel Bid", "Passive Cancel Ask"]
    sort!(integratedIntensities[1])
    quantiles = map(i -> quantile(Exponential(1), i / length(integratedIntensities[1])), 1:length(integratedIntensities[1]))
	qqPlot = plot([quantiles quantiles], [integratedIntensities[1] quantiles], seriestype = [:scatter :line], xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", label = [labels[1] ""], marker = (4, colors[1], stroke(colors[1]), 0.7), linecolor = :black, legendfontsize = 5, legend = :topleft, xlim = (0, 9), ylim = (0, 9))
	LBTemp = LjungBoxTest(integratedIntensities[1], length(integratedIntensities[1]) - 1, 210) # Ljung-Box - H_0 = independent
	KSTemp = ExactOneSampleKSTest(integratedIntensities[1], Exponential(1)) # Kolmogorov-Smirnov - H_0 = exponential
	LBTest[1, 1] = round(LBTemp.Q, digits = 5); LBTest[1, 2] = round(pvalue(LBTemp), digits = 5)
	KSTest[1, 1] = round(KSTemp.δ, digits = 5); KSTest[1, 2] = round(pvalue(KSTemp, tail = :both), digits = 5)
	for m in 2:dimension
		sort!(integratedIntensities[m])
	    quantiles = map(i -> quantile(Exponential(1), i / length(integratedIntensities[m])), 1:length(integratedIntensities[m]))
		plot!(qqPlot, [quantiles quantiles], [integratedIntensities[m] quantiles], seriestype = [:scatter :line], label = [labels[m] ""], marker = (4, colors[m], stroke(colors[m]), 0.7), linecolor = :black)
		LBTemp = LjungBoxTest(integratedIntensities[m], length(integratedIntensities[m]) - 1, 210) # Ljung-Box - H_0 = independent
		KSTemp = ExactOneSampleKSTest(integratedIntensities[m], Exponential(1)) # Kolmogorov-Smirnov - H_0 = exponential
		LBTest[m, 1] = round(LBTemp.Q, digits = 5); LBTest[m, 2] = round(pvalue(LBTemp), digits = 5)
		KSTest[m, 1] = round(KSTemp.δ, digits = 5); KSTest[m, 2] = round(pvalue(KSTemp, tail = :both), digits = 5)
	end
	savefig(qqPlot, string("Figures/HawkesQQPlot", model, ".", format))
	return LBTest, KSTest
end
#---------------------------------------------------------------------------------------------------
