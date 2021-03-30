#=
StylizedFacts:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Plot the stylized facts of HFT data for different time resolutions
- Structure:
    1. Log return sample distributions for different time resolutions
    2. Log-return and absolute log-return autocorrelation
    3. Trade sign autocorrealtion
    4. Trade inter-arrival time distribution
    5. Extreme log-return percentile distribution for different time resolutions
    6. Intraday statistics
TODO: Insert plot annotations for the values of α when fitting power laws
=#
using Distributions, CSV, Plots, DataFrames, StatsPlots, Dates, StatsBase, LaTeXStrings
clearconsole()
#---------------------------------------------------------------------------------------------------

#----- Log return sample distributions for different time resolutions -----#
function LogReturnDistribution(resolution::Symbol, cummulative::Bool = false)
    # Obtain log-returns of price series
    data = resolution == :TickbyTick ? CSV.File("L1LOB.csv", missingstring = "missing") |> DataFrame : CSV.File(string("MicroPrice ", resolution, " Bars.csv"), missingstring = "missing")
    logReturns = resolution == :TickbyTick ? diff(log.(filter(x -> !ismissing(x), data[:MicroPrice]))) : diff(log.(filter(x -> !ismissing(x), data[:Close])))
    # Fit theoretical distributions
    theoreticalDistribution1 = fit(Normal, logReturns)
    theoreticalDistribution2 = fit(InverseGaussian, logReturns)
    if cummulative
        # Plot empirical distribution
        empiricalDistribution = histogram(logReturns, normalize = :cdf, fillcolor = :blue, linecolor = :blue, xlabel = "Log returns", ylabel = "Density", label = "Empirical", legendtitle = "Distribution")
        # Plot fitted theoretical distributions
        plot!(empiricalDistribution, [cdf(theoreticalDistribution1, logReturns), cdf(theoreticalDistribution2, logReturns)], linecolor = [:green :purple], label = ["Fitted Normal" "Fitted Inverse Gaussian"])
        savefig(empiricalDistribution, string("Log-Return Cummulative Distribution - ", resolution, ".pdf"))
    else
        # Plot empirical distribution
        empiricalDistribution = histogram(logReturns, normalize = :pdf, fillcolor = :blue, linecolor = :blue, xlabel = "Log returns", ylabel = "Density", label = "Empirical", legendtitle = "Distribution")
        # Plot fitted theoretical distributions
        plot!(empiricalDistribution, [theoreticalDistribution1, theoreticalDistribution2], linecolor = [:green :purple], label = ["Fitted Normal" "Fitted Inverse Gaussian"])
        qqplot!(empiricalDistribution, Normal, logReturns, xlabel = "Normal theoretical quantiles", ylabel = "Sample quantiles", linecolor = :black, marker = (:blue, stroke(:blue)), legend = false, inset = (1, bbox(0.2, 0.1, 0.33, 0.33, :top)), subplot = 2)
        savefig(empiricalDistribution, string("Log-Return Distribution - ", resolution, ".pdf"))
    end
end
#---------------------------------------------------------------------------------------------------

#----- Log-return and absolute log-return autocorrelation -----#
function LogReturnAutocorrelation(lag::Int64)
    # Obtain log-returns
    data = CSV.File("L1LOB.csv", missingstring = "missing") |> DataFrame
    logReturns = diff(log.(filter(x -> !ismissing(x), data.MicroPrice)))
    # Calculate autocorrelations
    autoCorr = autocor(logReturns, 1:lag; demean = false)
    absAutoCorr = autocor(abs.(logReturns), 1:lag; demean = false)
    # Plot
    autoCorrPlot = plot(autoCorr, seriestype = :sticks, legend = false, xlabel = "Lag", ylabel = "Autocorrelation")
    plot!(autoCorrPlot, [1.96 / sqrt(length(logReturns)), -1.96 / sqrt(length(logReturns))], seriestype = :hline, line = (:dash, :black, 1))
    plot!(autoCorrPlot, absAutoCorr, seriestype = :sticks, legend = false, xlabel = "Lag", ylabel = "Autocorrelation", inset = (1, bbox(0.62, 0.55, 0.33, 0.33, :top)), subplot = 2)
    savefig(autoCorrPlot, "Log-Return Autocorrelation.pdf")
end
#---------------------------------------------------------------------------------------------------

#----- Trade sign autocorrealtion -----#
function TradeSignAutocorrelation(lag::Int64)
    # Extract trade signs
    data = CSV.File("L1LOB.csv", missingstring = "missing")
    tradeSigns = filter(x -> x.Type == :MO, data).Side
    # Calculate autocorrelations
    autoCorr = autocor(tradeSigns, 1:lag; demean = false)
    # Plot
    autoCorrPlot = plot(autoCorr, seriestype = :sticks, legend = false, xlabel = "Lag", ylabel = "Autocorrelation")
    plot!(autoCorrPlot, [quantile(Normal(), (1 + 0.95) / 2) / sqrt(length(tradeSigns)), quantile(Normal(), (1 - 0.95) / 2) / sqrt(length(tradeSigns))], seriestype = :hline, line = (:dash, :black, 1))
    plot!(autoCorrPlot, autoCorr, xscale = :log10, inset = (1, bbox(0.58, 0.0, 0.4, 0.4)), subplot = 2, legend = false, xlabel = "Lag " * L"(\log_{10})", ylabel = "Autocorrelation", linecolor = :black)
    savefig(autoCorrPlot, "Trade-Sign Autocorrelation.pdf")
end
#---------------------------------------------------------------------------------------------------

#----- Trade inter-arrival time distribution -----#
function InterArrivalTimeDistribution()
    # Extract inter-arrival times
    data = CSV.File("L1LOB.csv", missingstring = "missing")
    interArrivals = filter(x -> x.Type == :MO, data) |> y -> diff(y.DateTime) |> z -> Dates.value.(round.(z, Dates.Millisecond))
    # Estimate power-law distribution parameters
    xₘᵢₙ = minimum(interArrivals)
    α = 1 + length(interArrivals) / sum(log.(interArrivals ./ xₘᵢₙ))
    # Extract theoretical quantiles
    theoreticalQuantiles = map(i -> (1 - (i / length(interArrivals))) ^ (-1 / (α - 1)) * xₘᵢₙ, 1:length(interArrivals))
    theoreticalDistribution1 = fit(Exponential, interArrivals)
    theoreticalDistribution2 = fit(Weibull, interArrivals)
    # Plot
    logDistribution = histogram(interArrivals, normalize = :pdf, linecolor = :blue, fillcolor = :blue, yscale = :log10, xlabel = "Inter-arrival time (milliseconds)", ylabel = "Log Density", legend = false)
    plot!(logDistribution, [theoreticalDistribution1, theoreticalDistribution2], linecolor = [:green :purple], label = ["Fitted Exponential" "Fitted Weibull"])
    plot!(logDistribution, [theoreticalQuantiles theoreticalQuantiles], [interArrivals theoreticalQuantiles], seriestype = [:scatter :line], inset = (1, bbox(0.6, 0.03, 0.34, 0.34, :top)), subplot = 2, legend = :none, xlabel = "Power-Law Theoretical Quantiles", ylabel = "Sample Quantiles", linecolor = :black, markercolor = :blue, markerstrokecolor = :blue, scale = :log10)
    savefig(logDistribution, "Trade Log-Inter-Arrival Distribution.pdf")
end
#---------------------------------------------------------------------------------------------------

#----- Extreme log-return percentile distribution for different time resolutions -----#
function ExtremeLogReturnPercentileDistribution(resolution::Symbol, side::Symbol)
    # Obtain log-returns of price series
    if resolution == :TickbyTick
        data = CSV.File("L1LOB.csv", missingstring = "missing") |> DataFrame
        logReturns = diff(log.(filter(x -> !ismissing(x), data[:MicroPrice])))
    else
        data = CSV.File(string("MicroPrice ", resolution, " Bars.csv"), missingstring = "missing") |> DataFrame
        logReturns = diff(log.(filter(x -> !ismissing(x), data[:Close])))
    end
    # Extract extreme empirical quantiles
    observations = side == :Upper ? logReturns[findall(x -> x >= quantile(logReturns, 0.95), logReturns)] : -logReturns[findall(x -> x <= quantile(logReturns, 0.05), logReturns)]
    # Estimate power-law distribution parameters
    xₘᵢₙ = minimum(observations)
    α = 1 + length(observations) / sum(log.(observations ./ xₘᵢₙ))
    # Extract theoretical quantiles
    theoreticalQuantiles = map(i -> (1 - (i / length(observations))) ^ (-1 / (α - 1)) * xₘᵢₙ, 1:length(observations))
    # Plot
    extremePercentileDistributionPlot = density(observations, linecolor = :blue, xlabel = "Log return Extreme", side, " percentiles", ylabel = "Density", legend = :none)
    plot!(extremePercentileDistributionPlot, [theoreticalQuantiles theoreticalQuantiles], [observations theoreticalQuantiles], seriestype = [:scatter :line], inset = (1, bbox(0.6, 0.03, 0.34, 0.34, :top)), subplot = 2, legend = :none, xlabel = "Power-Law Theoretical Quantiles", ylabel = "Sample Quantiles", linecolor = :black, markercolor = :blue, markerstrokecolor = :blue, scale = :log10)
    savefig(extremePercentileDistributionPlot, string("Extreme", side, " Log-Return Percentiles Distribution - ", resolution,".pdf"))
end
#---------------------------------------------------------------------------------------------------
