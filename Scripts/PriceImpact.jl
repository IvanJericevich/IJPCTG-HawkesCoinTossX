#=
PriceImpact:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Analyze price-impact data and construct master curves
- Structure:
    1. Extract price-impact data
    2. Objective for master curve calibration
    3. Extract master curve data
- Examples
    LogReturnDistribution(:TickbyTick; lobFile = "Model2L1LOB", format = "png")
    LogReturnAutocorrelation(50, lobFile = "Model2L1LOB", format = "png")
    TradeSignAutocorrelation(20, lobFile = "Model2L1LOB", format = "png")
    InterArrivalTimeDistribution("Model2L1LOB"; format = "png")
    ExtremeLogReturnPercentileDistribution(:TickbyTick, :Upper; lobFile = "Model2L1LOB", format = "png")
- TODO: Insert plot annotations for the values of master curve parameters
- TODO: Change font sizes
=#
using Statistics, DataFrames, CSV, Plots, Optim
clearconsole()
#---------------------------------------------------------------------------------------------------

#----- Extract price-impact data -----#
#=
Function:
    - Extract raw impacts and normalized volumes
    - This relates to a single security/model over multiple simulation periods
Arguments:
    - files = L1LOB file names that relate to a single model/security
Output:
    - Buyer-initiated and seller-initiated price impacts and nomralized volumes as well as the average daily value traded
=#
function PriceImpact(files::Vector{String}) # REVIEW: We are only doing one day so this will only be one file. This is also only for one security. Calculate price impact for a single security ofr multiple days
    data = Dict(i => (CSV.File(string("Data/", files[i], ".csv"), types = Dict(:Type => Symbol), missingstring = "missing") |> DataFrame) for i in 1:length(files))
    tradeIndeces = [Vector{Int64}() for _ in 1:length(files)]; totalTradeCount = 0; dailyValue = zeros(Int64, length(files)) # Initialization
    for (k, v) in data
        tradeIndeces[k] = findall(x -> x == :MO, v.Type)
        totalTradeCount += length(tradeIndeces[k])
        dailyValue[k] = sum(v.Price[tradeIndeces[k]] * v.Volume[tradeIndeces[k]])
    end
    averageDailyValue = mean(dailyValue)
    buyerInitiated = DataFrame(Impact = Vector{Float64}(), NormalizedVolume = Vector{Float64}()); sellerInitiated = DataFrame(Impact = Vector{Float64}(), NormalizedVolume = Vector{Float64}())
    for (k, v) in data
        dayVolume = sum(v.Volume[tradeIndeces[k]])
        for index in tradeIndeces[k]
            midPriceBeforeTrade = data.MidPrice[index - 1]; midPriceAfterTrade = data.MidPrice[index + 1]
            Δp = log(midPriceAfterTrade) - log(midPriceBeforeTrade)
            ω = (data.Volume[index] / dayVolume) * (totalTradeCount / length(files))
            v.Side[index] == -1 ? push!(buyerInitiated, (Δp, ω)) : push!(sellerInitiated, (-Δp, ω))
        end
    return buyerInitiated, sellerInitiated, averageDailyValue
end
#=
function PlotPriceImpact(data::DataFrame)
    ω = zeros(Float64, 21); Δp = zeros(Float64, 21)
    volumeBins = 10 .^ (range(-3, 1, length = 21))
    for i in 2:length(volumeBins)
        binIndeces = findall(x -> volumeBins[i - 1] < x <= x[i], data.NormalizedVolume)
        ω[i - 1] = data.NormalizedVolume[binIndeces]; Δp[i - 1] = data.Impact[binIndeces]
    end
    plot(ω, Δp, seriestype = [:scatter, :line], scale = :log10, marker = (:blue, stroke(:blue)), linecolor = :blue, label = ["Stock 1" ""], xlabel = "\\omega", ylabel = "\\Delta p")
end
=#
#---------------------------------------------------------------------------------------------------

#----- Objective for master curve calibration -----#
function Average2DVariance(parameters::Vector{Type}, priceImpacts::Vector{DataFrame}, averageDailyValues::Vector{Type}; low::Type = -1, up::Type = 1) where Type <: Real
    δ = parameters[1]; γ = parameters[2]
    volumeBins = 10 .^ (range(low, up, length = 21))
    μₓ = zeros(Type, 20); σₓ = zeros(Type, 20)
    μᵧ = zeros(Type, 20); σᵧ = zeros(Type, 20)
    for i in 2:length(bins)
        Δp = zeros(Type, length(priceImpacts)); ω = zeros(Type, length(priceImpacts)) # Initialize mean impact and normalized volume for this bin
        for j in 1:length(priceImpacts)
            priceImpactData = priceImpacts[j]
            C = averageDailyValues[j]
            bin = findall(x -> volumeBins[i - 1] < x <= volumeBins[i], priceImpactData.NormalizedVolume)
            if !isempty(bin) # If there are no normalized volumes in this bin then values remains zero
                Δp[j] = mean(priceImpactData.Impact[bin]) * (C ^ γ)
                ω[j] = mean(priceImpactData.NormalizedVolume[bin]) / (C ^ δ)
            end
        end
        filter!(!iszero, Δp); filter!(!iszero, ω)
        μᵧ[i - 1] = mean(Δp); σᵧ[i - 1] = std(Δp)
        μₓ[i - 1] = mean(ω); σₓ[i - 1] = std(ω)
    end
    return sum((σₓ ./ μₓ) .^ 2 + (σᵧ ./ μᵧ) .^ 2) / 20
end
#---------------------------------------------------------------------------------------------------

#----- Extract master curve data -----#
#=
Function:
    - Bin and optimally scale price impact data to collapse price impact curves into a single master curve
    - This relates to multiple securities/models over multiple simulation periods
Arguments:
    - data = vector of buyer/seller initiated price impact data for multiple simulation periods. Each element corresponds to a different model/security
Output:
    - An array of buyer/seller intiated master curves. Columns are securities/models; rows are bins
=#
function MasterCurves(data::Vector{DataFrame}, averageDailyValues::Vector{Float64}; low::Type = -1, up::Type = 1)
    parameters = optimize(θ -> Average2DVariance(θ, buyerInitiated, averageDailyValues), [0.3, 0.3]) |> minimizer
    δ = parameters[1]; γ = parameters[2]
    volumeBins = 10 .^ (range(low, up, length = 21))
    Δp = zeros(Float64, 20, length(data)); ω = zeros(Float64, 20, length(data))
    for j in 1:length(data)
        C = averageDailyValues[j]
        for i in 2:length(volumeBins)
            bin = findall(x -> volumeBins[i - 1] < x <= volumeBins[i], data[j].NormalizedVolume)
            if !isempty(bin)
                Δp[i - 1, j] = mean(data[j].Impact[bin]) * (C ^ γ)
                ω[i - 1, j] = mean(data[j].NormalizedVolume[bin]) / (C ^ δ)
            end
        end
    end
    return ω, Δp
end
#=
Typically there would be multiple tickers each with a single dataframe with multiple dates
buyerInitiated = [DataFrame() for _ in 1:length(files)]
sellerInitiated = [DataFrame() for _ in 1:length(files)]
averageDailyValues = zeros(length(files))
for i in files
    buyerInitiated[i], sellerInitiated[i], averageDailyValue[i] = PriceImpact(files[i]])
end
=#
#---------------------------------------------------------------------------------------------------
