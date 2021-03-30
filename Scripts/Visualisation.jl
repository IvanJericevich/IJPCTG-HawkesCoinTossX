#=
Visualisation:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Basic visualisations of HFT time series
- Structure:
    1. Data preparation
    2. Plot simulation results
=#
using Dates, DataFrames, CSV, Plots, Statistics
clearconsole()
#---------------------------------------------------------------------------------------------------

#----- Data preparation -----#
limitOrders = CSV.File("Data/OrdersSubmitted_2.csv", drop = [:SecurityId, :OrderId], types = Dict(:ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64, :Side => Symbol), dateformat = "yyyy-mm-dd HH:MM:SS.s") |> DataFrame |> x -> filter(y -> y.Price != 0, x)
marketOrders = CSV.File("Data/Trades_2.csv", drop = [:OrderId], types = Dict(:ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64), dateformat = "yyyy-mm-dd HH:MM:SS.s") |> DataFrame
marketOrders.ContraSide = limitOrders.Side[indexin(marketOrders.ClientOrderId, limitOrders.ClientOrderId)] # Extract MO contra side
l1lob = CSV.File("Data/Model2L1LOB.csv", types = Dict(:DateTime => DateTime, :Price => Int64, :Volume => Int64, :Type => Symbol, :Side => Int64, :MidPrice => Float64, :MicroPrice => Float64, :Spread => Float64), missingstring = "missing") |> DataFrame |> x -> filter(y -> !ismissing(y.MidPrice), x)
#---------------------------------------------------------------------------------------------------

#----- Plot simulation results -----#
asks = filter(x -> x.Side == :Sell && x.Volume != 0, limitOrders); bids = filter(x -> x.Side == :Buy && x.Volume != 0, limitOrders)
sells = filter(x -> x.ContraSide == :Buy, marketOrders); buys = filter(x -> x.ContraSide == :Sell, marketOrders)
cancelAsks = filter(x -> x.Side == :Sell && x.Volume == 0, limitOrders); cancelBids = filter(x -> x.Side == :Buy && x.Volume == 0, limitOrders)
bubblePlot = plot(Time.(asks.DateTime), asks.Price, seriestype = :scatter, marker = (:red, stroke(:red), 5 * (asks.Volume / mean(asks.Volume)), 0.7), label = "Ask (LO)", xlabel = "Time", ylabel = "Price (ticks)", legend = :topleft, legendfontsize = 5, xrotation = 30)
plot!(bubblePlot, Time.(bids.DateTime), bids.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), 5 * (bids.Volume / mean(bids.Volume)), 0.7), label = "Bid (LO)")
plot!(bubblePlot, Time.(sells.DateTime), sells.Price, seriestype = :scatter, marker = (:red, stroke(:red), :utriangle, 5 * (sells.Volume / mean(sells.Volume)), 0.7), label = "Sell (MO)")
plot!(bubblePlot, Time.(buys.DateTime), buys.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), :utriangle, 5 * (buys.Volume / mean(buys.Volume)), 0.7), label = "Buy (MO)")
plot!(bubblePlot, Time.(cancelAsks.DateTime), cancelAsks.Price, seriestype = :scatter, marker = (:red, stroke(:red), :xcross, 0.7), label = "Cancel Ask (LO)")
plot!(bubblePlot, Time.(cancelBids.DateTime), cancelBids.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), :xcross, 0.7), label = "Cancel Bid (LO)")
plot!(bubblePlot, Time.(l1lob.DateTime), l1lob.MidPrice, seriestype = :line, linecolor = :black, label = "Mid-price")
plot!(bubblePlot, Time.(l1lob.DateTime), l1lob.MicroPrice, seriestype = :line, linecolor = :green, label = "Micro-price")
savefig(bubblePlot, "Figures/Model2BubblePlot.pdf")
#---------------------------------------------------------------------------------------------------
