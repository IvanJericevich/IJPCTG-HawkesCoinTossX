#=
DataCleaning:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Clean CoinTossX simulated data into L1LOB and OHLCV data for stylized fact analysis as well as visualisation of HFT time-series
- Structure:
    1. Data preparation
    2. Supplementary functions
    3. Clean raw data into L1LOB and OHLCV data
    4. Plot simulation results
- TODO: Add a column to the orders dataframe for whether the order was aggressive or not
=#
using CSV, DataFrames, Dates, Plots, Statistics
clearconsole()
PrepareData("OrdersSubmitted_2", "Trades_2") |> x -> CleanData(x; allowCrossing = false)
VisualiseSimulation(5, "OrdersSubmitted_2", "Trades_2", "Model2L1LOB")
#---------------------------------------------------------------------------------------------------

#----- Data preparation -----#
function PrepareData(ordersSubmitted::String, trades::String)
    rawOrders = CSV.File(string("Data/", ordersSubmitted, ".csv"), drop = [:SecurityId, :OrderId], types = Dict(:ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64, :Side => Symbol), dateformat = "yyyy-mm-dd HH:MM:SS.s") |> DataFrame |> x -> filter(y -> y.Price != 0, x)
    limitOrders = filter(x -> x.ClientOrderId > 0, rawOrders) # Limit orders have positive ID
    cancelOrders = filter(x -> x.ClientOrderId < 0, rawOrders) # Cancel orders have negative ID
    marketOrders = CSV.File(string("Data/", trades, ".csv"), drop = [:OrderId], types = Dict(:ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64), dateformat = "yyyy-mm-dd HH:MM:SS.s") |> DataFrame
    cancelOrders.ClientOrderId .*= -1 # Make cancel IDs positive again
    cancelOrders[:, [:Price, :Volume]] = limitOrders[indexin(cancelOrders.ClientOrderId, limitOrders.ClientOrderId), [:Price, :Volume]] # Associate order cancels with their corresponding LO volume and price to make cleaning easier
    marketOrders.Side = limitOrders.Side[indexin(marketOrders.ClientOrderId, limitOrders.ClientOrderId)] # Extract MO contra side
    limitOrders.Type = fill(:LO, nrow(limitOrders)); cancelOrders.Type = fill(:OC, nrow(cancelOrders)); marketOrders.Type = fill(:MO, nrow(marketOrders)) # Assign order types
    orders = outerjoin(limitOrders, marketOrders, cancelOrders, on = [:ClientOrderId, :DateTime, :Price, :Volume, :Type, :Side]) # Combine everything
    rename!(orders, :ClientOrderId => :OrderId)
    sort!(orders, :DateTime)
    return orders
end
#---------------------------------------------------------------------------------------------------

#----- Supplementary functions -----#
function MidPrice(bestBid::NamedTuple, bestAsk::NamedTuple)::Union{Missing, Float64}
    return (isempty(bestBid) || isempty(bestAsk)) ? missing : (bestBid.Price + bestAsk.Price) / 2
end
function MicroPrice(bestBid::NamedTuple, bestAsk::NamedTuple)::Union{Missing, Float64}
    return (isempty(bestBid) || isempty(bestAsk)) ? missing : (bestBid.Price * bestBid.Volume + bestAsk.Price * bestAsk.Volume) / (bestBid.Volume + bestAsk.Volume)
end
function Spread(bestBid::NamedTuple, bestAsk::NamedTuple)::Union{Missing, Float64}
    return (isempty(bestBid) || isempty(bestAsk)) ? missing : abs(bestBid.Price - bestAsk.Price)
end
#---------------------------------------------------------------------------------------------------

#----- Clean raw data into L1LOB and OHLCV data -----#
function CleanData(orders::DataFrame; visualise::Bool = false, allowCrossing::Bool = false)
    open("Data/Model2L1LOB.csv", "w") do file
        println(file, "DateTime,Price,Volume,Type,Side,MidPrice,MicroPrice,Spread")
        bids = Dict{Int64, Tuple{Int64, Int64}}(); asks = Dict{Int64, Tuple{Int64, Int64}}() # Both sides of the entire LOB are tracked with keys corresponding to orderIds
        bestBid = NamedTuple(); bestAsk = NamedTuple() # Current best bid/ask is stored in a tuple (Price, vector of Volumes, vector of OrderIds) and tracked
        animation = visualise ? Animation() : nothing
        Juno.progress() do id
            for i in 1:nrow(orders) # Iterate through all orders
                order = orders[i, :]
                #-- Limit Orders --#
                if order.Type == :LO
                    if order.Side == :Buy # Buy limit order
                        if isempty(bids) || isempty(bestBid) # If the dictionary is empty, this order automatically becomes best
                            bestBid = (Price = order.Price, Volume = order.Volume, OrderId = [order.OrderId]) # New best is created
                            midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                            println(file, string(order.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",LO,1,", midPrice, ",", microPrice, ",", spread))
                        else # Otherwise find the best
                            if order.Price > bestBid.Price # Change best if price of current order better than the best
                                if !isempty(bestAsk) && order.Price >= bestAsk.Price # Bid crosses best ask => limit order becomes effective market order (to avoid an error first check if the otherside isn't empty)
                                    allowCrossing ? (println(string("Order ", order.OrderId, " crossed the spread")); continue) : error("Negative spread - bid has crossed ask at order " * string(order.OrderId)) # Either disallow crossing or treat crossed order as effective market order (in which case ignore the limit order and only process subsequent market order)
                                end
                                bestBid = (Price = order.Price, Volume = order.Volume, OrderId = [order.OrderId]) # New best is created
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(order.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",LO,1,", midPrice, ",", microPrice, ",", spread))
                            elseif order.Price == bestBid.Price # Add the new order's volume and orderid to the best if they have the same price
                                bestBid = (Price = bestBid.Price, Volume = bestBid.Volume + order.Volume, OrderId = vcat(bestBid.OrderId, order.OrderId)) # Best is ammended by adding volume to best and appending the order id
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(order.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",LO,1,", midPrice, ",", microPrice, ",", spread))
                            end # Otherwise the L1LOB hasn't changed so do nothing
                        end
                        push!(bids, order.OrderId => (order.Price, order.Volume)) # New order is always pushed to LOB dictionary only after best is processed
                    else # Sell limit order
                        if isempty(asks) || isempty(bestAsk) # If the dictionary is empty, this order automatically becomes best
                            bestAsk = (Price = order.Price, Volume = order.Volume, OrderId = [order.OrderId]) # New best is created
                            midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                            println(file, string(order.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",LO,-1,", midPrice, ",", microPrice, ",", spread))
                        else # Otherwise find the best
                            if order.Price < bestAsk.Price # Change best if price of current order better than the best
                                if !isempty(bestBid) && order.Price <= bestBid.Price # Ask crosses best bid => limit order becomes effective market order (to avoid an error first check if the otherside isn't empty)
                                    allowCrossing ? (println(string("Order ", order.OrderId, " crossed the spread")); continue) : error("Negative spread - ask has crossed bid at order " * string(order.OrderId)) # Either disallow crossing or treat crossed order as effective market order (in which case ignore the limit order and only process subsequent market order since trade is printed in Trades.csv and will be handled anyway)
                                end
                                bestAsk = (Price = order.Price, Volume = order.Volume, OrderId = [order.OrderId]) # New best is created
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(order.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",LO,-1,", midPrice, ",", microPrice, ",", spread))
                            elseif order.Price == bestAsk.Price # Add the new order's volume and orderid to the best if they have the same price
                                bestAsk = (Price = bestAsk.Price, Volume = bestAsk.Volume + order.Volume, OrderId = vcat(bestAsk.OrderId, order.OrderId)) # Best is ammended by adding volume to best and appending the order id
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(order.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",LO,-1,", midPrice, ",", microPrice, ",", spread))
                            end # Otherwise the L1LOB hasn't changed so do nothing
                        end
                        push!(asks, order.OrderId => (order.Price, order.Volume)) # New order is always pushed to LOB dictionary only after best is processed
                    end
                #-- Market Orders --#
                elseif order.Type == :MO # Market order always affects the best
                    if order.Side == :Buy # Trade was buyer-initiated (Sell MO)
                        # REVIEW: If the OrderId references itself then a Null Pointer will occur, so set the contraOrder to the price-time priority best (clientOrderId is required to be in the same order as time)
                        if !haskey(bids, order.OrderId) # Trade ID is not present in LOB
                            order.OrderId = minimum(keys(bids))
                            contraOrder = bids[order.OrderId]
                            order.Price = first(contraOrder)
                            crossedOrder = orders[i - 1, :] # Get the original limit order that crossed the spread
                            crossedOrder.Volume != order.Volume ? push!(bids, crossedOrder.OrderId => (order.Price, crossedOrder.Volume - order.Volume)) : nothing # Remove the executed quantity from the LO and push the remaining volume to the LOB
                        else
                            contraOrder = bids[order.OrderId]
                        end
                        #contraOrder = bids[order.OrderId]
                        println(file, string(order.DateTime, ",", order.Price, ",", order.Volume, ",MO,-1,missing,missing,missing")) # Sell trade is printed
                        if order.Volume == bestBid.Volume # Trade filled best - remove from LOB, and update best
                            delete!(bids, order.OrderId) # Remove the order from the LOB
                            if !isempty(bids) # If the LOB is non empty find the best
                                bestPrice = maximum(first.(collect(values(bids)))) # Find the new best price
                                indeces = findall(x -> first(x) == bestPrice, bids) # Find the order ids of the best
                                bestBid = (Price = bestPrice, Volume = sum(last(bids[i]) for i in indeces), OrderId = indeces) # Update the best
                            else # If the LOB is empty remove best
                                bestBid = NamedTuple()
                            end
                        else # Trade partially filled best
                            if order.Volume == contraOrder[2] # Trade filled contra order - remove order from LOB, remove order from best, and update best
                                delete!(bids, order.OrderId)
                                bestBid = (Price = bestBid.Price, Volume = bestBid.Volume - order.Volume, OrderId = setdiff(bestBid.OrderId, order.OrderId))
                            else # Trade partially filled contra order - update LOB, update best
                                bids[order.OrderId] = (contraOrder[1], contraOrder[2] - order.Volume)
                                bestBid = (Price = bestBid.Price, Volume = bestBid.Volume - order.Volume, OrderId = bestBid.OrderId)
                            end
                        end
                        midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                        !isempty(bestBid) ? println(file, string(order.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",LO,1,", midPrice, ",", microPrice, ",", spread)) : println(file, string(order.DateTime, ",missing,missing,LO,1,missing,missing,missing"))
                    else # Trade was seller-initiated (Buy MO)
                        # REVIEW: If the OrderId references itself then a Null Pointer will occur, so set the contraOrder to the price-time priority best (clientOrderId is required to be in the same order as time)
                        if !haskey(asks, order.OrderId) # Trade ID is not present in LOB
                            order.OrderId = minimum(keys(asks))
                            contraOrder = asks[order.OrderId]
                            order.Price = first(contraOrder)
                            crossedOrder = orders[i - 1, :] # Get the original limit order that crossed the spread
                            crossedOrder.Volume != order.Volume ? push!(asks, crossedOrder.OrderId => (order.Price, crossedOrder.Volume - order.Volume)) : nothing # Remove the executed quantity from the LO and push the remaining volume to the LOB
                        else
                            contraOrder = asks[order.OrderId]
                        end
                        #contraOrder = asks[order.OrderId]
                        # contraOrder = haskey(asks, order.OrderId) ? asks[order.OrderId] : asks[minimum(keys(asks))] # REVIEW: If the OrderId references itself then a Null Pointer will occur, so set the contraOrder to the price-time priority best (clientOrderId is in the same order as time)
                        println(file, string(order.DateTime, ",", order.Price, ",", order.Volume, ",MO,1,missing,missing,missing")) # Buy trade is printed
                        if order.Volume == bestAsk.Volume # Trade filled best - remove from LOB, and update best
                            delete!(asks, order.OrderId) # Remove the order from the LOB
                            if !isempty(asks) # If the LOB is non empty find the best
                                bestPrice = maximum(first.(collect(values(asks)))) # Find the new best price
                                indeces = findall(x -> first(x) == bestPrice, asks) # Find the order ids of the best
                                bestAsk = (Price = bestPrice, Volume = sum(last(asks[i]) for i in indeces), OrderId = indeces) # Update the best
                            else # If the LOB is empty remove best
                                bestAsk = NamedTuple()
                            end
                        else # Trade partially filled best
                            if order.Volume == contraOrder[2] # Trade filled contra order - remove order from LOB, remove order from best, and update best
                                delete!(asks, order.OrderId)
                                bestAsk = (Price = bestAsk.Price, Volume = bestAsk.Volume - order.Volume, OrderId = setdiff(bestAsk.OrderId, order.OrderId))
                            else # Trade partially filled contra order - update LOB, update best
                                asks[order.OrderId] = (contraOrder[1], contraOrder[2] - order.Volume)
                                bestAsk = (Price = bestAsk.Price, Volume = bestAsk.Volume - order.Volume, OrderId = bestAsk.OrderId)
                            end
                        end
                        midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                        !isempty(bestAsk) ? println(file, string(order.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",LO,-1,", midPrice, ",", microPrice, ",", spread)) : println(file, string(order.DateTime, ",missing,missing,LO,-1,missing,missing,missing"))
                    end
                #-- Cancel Orders --#
                else
                    if order.Side == :Buy # Cancel buy limit order
                        delete!(bids, order.OrderId) # Remove the order from the LOB
                        if order.OrderId in bestBid.OrderId # Cancel hit the best
                            activeL1Orders = setdiff(bestBid.OrderId, order.OrderId) # Find the remaining level 1 order ids
                            if !isempty(activeL1Orders) # Cancel did not empty the best - remove order from best and update best
                                bestBid = (Price = bestBid.Price, Volume = bestBid.Volume - order.Volume, OrderId = activeL1Orders)
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(order.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",OC,1,", midPrice, ",", microPrice, ",", spread))
                            else # Cancel emptied the best
                                if !isempty(bids) # Orders still remain in the buy side LOB - update best
                                    bestPrice = maximum(first.(collect(values(bids)))) # Find the new best price
                                    indeces = findall(x -> first(x) == bestPrice, bids) # Find the order ids of the best
                                    bestBid = (Price = bestPrice, Volume = sum(last(bids[i]) for i in indeces), OrderId = indeces) # Update the best
                                    midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                    println(file, string(order.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",OC,1,", midPrice, ",", microPrice, ",", spread))
                                else # The buy side LOB was emptied - update best
                                    bestBid = NamedTuple()
                                    println(file, string(order.DateTime, ",missing,missing,OC,1,missing,missing,missing"))
                                end
                            end
                        end
                    else # Cancel sell limit order
                        delete!(asks, order.OrderId) # Remove the order from the LOB
                        if order.OrderId in bestAsk.OrderId # Cancel hit the best
                            activeL1Orders = setdiff(bestAsk.OrderId, order.OrderId) # Find the remaining level 1 order ids
                            if !isempty(activeL1Orders) # Cancel did not empty the best - remove order from best and update best
                                bestAsk = (Price = bestAsk.Price, Volume = bestAsk.Volume - order.Volume, OrderId = activeL1Orders)
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(order.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",OC,-1,", midPrice, ",", microPrice, ",", spread))
                            else # Cancel emptied the best
                                if !isempty(asks) # Orders still remain in the sell side LOB - update best
                                    bestPrice = maximum(first.(collect(values(asks)))) # Find the new best price
                                    indeces = findall(x -> first(x) == bestPrice, asks) # Find the order ids of the best
                                    bestAsk = (Price = bestPrice, Volume = sum(last(asks[i]) for i in indeces), OrderId = indeces) # Update the best
                                    midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                    println(file, string(order.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",OC,-1,", midPrice, ",", microPrice, ",", spread))
                                else # The sell side LOB was emptied - update best
                                    bestAsk = NamedTuple()
                                    println(file, string(order.DateTime, ",missing,missing,OC,-1,missing,missing,missing"))
                                end
                            end
                        end
                    end
                end
                if visualise # Visualisation
                    PlotLOBSnapshot(bids, asks)
                    frame(animation)
                end
                @info "Cleaning:" progress=(i / nrow(orders)) _id=id # Update progress
            end
            visualise ? gif(animation, "LOB.gif", fps = 3) : nothing
        end
        Juno.notification("Data cleaning complete"; kind = :Info, options = Dict(:dismissable => false))
    end
end
function PlotLOBSnapshot(bids::Dict{Int64, Tuple{Int64, Int64}}, asks::Dict{Int64, Tuple{Int64, Int64}})
    if !isempty(bids) && !isempty(asks)
        bidSnapshot = collect(values(bids)) |> x -> DataFrame(Price = first.(x), Volume = last.(x)) |> y -> groupby(y, :Price) |> z -> combine(z, :Volume => sum => :Volume); askSnapshot = collect(values(asks)) |> x -> DataFrame(Price = first.(x), Volume = last.(x)) |> y -> groupby(y, :Price) |> z -> combine(z, :Volume => sum => :Volume)
        plot(bidSnapshot.Price, bidSnapshot.Volume, seriestype = :bar, fillcolor = :blue, linecolor = :blue, label = "Bid", xlabel = "Price", ylabel = "Depth", legendtitle = "Side")
        plot!(askSnapshot.Price, askSnapshot.Volume, seriestype = :bar, fillcolor = :red, linecolor = :red, label = "Ask")
    elseif isempty(bids) && isempty(asks)
        plot(xlabel = "Price", ylabel = "Depth")
    elseif !isempty(bids)
        bidSnapshot = collect(values(bids)) |> x -> DataFrame(Price = first.(x), Volume = last.(x)) |> y -> groupby(y, :Price) |> z -> combine(z, :Volume => sum => :Volume)
        plot(bidSnapshot.Price, bidSnapshot.Volume, seriestype = :bar, fillcolor = :blue, linecolor = :blue, label = "Bid", xlabel = "Price", ylabel = "Depth", legendtitle = "Side")
    elseif !isempty(asks)
        askSnapshot = collect(values(asks)) |> x -> DataFrame(Price = first.(x), Volume = last.(x)) |> y -> groupby(y, :Price) |> z -> combine(z, :Volume => sum => :Volume)
        plot(askSnapshot.Price, askSnapshot.Volume, seriestype = :bar, fillcolor = :red, linecolor = :red, label = "Ask", xlabel = "Price", ylabel = "Depth", legendtitle = "Side")
    end
end
function OHLCV(orders, resolution)
    open(string("OHLCV.csv"), "w") do file
        println(file, "DateTime,MidOpen,MidHigh,MidLow,MidClose,MicroOpen,MicroHigh,MicroLow,MicroClose,Volume,VWAP")
        barTimes = orders.DateTime[1]:resolution:orders.DateTime[end]
        for t in 1:(length(barTimes) - 1)
            startIndex = searchsortedfirst(orders.DateTime, barTimes[i])
            endIndex = searchsortedlast(orders.dateTime, barTimes[i + 1])
            if !(startIndex >= endIndex)
                bar = orders[startIndex:endIndex, :]
                tradesBar = filter(x -> x.Type == "Trade", bar)
                midPriceOHLCV = string(bar.MidPrice[1], ",", maximum(bar.MidPrice), ",", minimum(bar.MidPrice), ",", bar.MidPrice[end])
                microPriceOHLCV = string(bar.MicroPrice[1], ",", maximum(bar.MicroPrice), ",", minimum(bar.MicroPrice), ",", bar.MicroPrice[end])
                vwap = !isempty(tradesBar) ? sum(tradesBar.Volume .* tradesBar.Price) / sum(tradesBar.Volume) : missing
                println(file, string(barTimes[i], ",", midPriceOHLCV, ",", microPriceOHLCV, ",", sum(bar.Volume), ",", vwap))
            end
        end
    end
end
#---------------------------------------------------------------------------------------------------

#----- Plot simulation results -----#
function VisualiseSimulation(scale::Int64, ordersSubmitted::String, trades::String, l1lob::String)
    orders = PrepareData(ordersSubmitted, trades)
    l1lob = CSV.File("Data/", l1lob, ".csv", types = Dict(:Type => Symbol), missingstring = "missing") |> DataFrame #> x -> filter(y -> !ismissing(y.MidPrice), x)
    asks = filter(x -> x.Type == :LO && x.Side == :Sell, orders); bids = filter(x -> x.Type == :LO && x.Side == :Buy, orders)
    sells = filter(x -> x.Type == :MO && x.Side == :Buy, orders); buys = filter(x -> x.Type == :MO && x.Side == :Sell, orders)
    cancelAsks = filter(x -> x.Type == :OC && x.Side == :Sell, orders); cancelBids = filter(x -> x.Type == :OC && x.Side == :Buy, orders)
    bubblePlot = plot(Time.(asks.DateTime), asks.Price, seriestype = :scatter, marker = (:red, stroke(:red), scale * (asks.Volume / mean(asks.Volume)), 0.7), label = "Ask (LO)", xlabel = "Time", ylabel = "Price (ticks)", legend = :topleft, legendfontsize = 5, xrotation = 30)
    plot!(bubblePlot, Time.(bids.DateTime), bids.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), scale * (bids.Volume / mean(bids.Volume)), 0.7), label = "Bid (LO)")
    plot!(bubblePlot, Time.(sells.DateTime), sells.Price, seriestype = :scatter, marker = (:red, stroke(:red), :utriangle, scale * (sells.Volume / mean(sells.Volume)), 0.7), label = "Sell (MO)")
    plot!(bubblePlot, Time.(buys.DateTime), buys.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), :utriangle, scale * (buys.Volume / mean(buys.Volume)), 0.7), label = "Buy (MO)")
    plot!(bubblePlot, Time.(cancelAsks.DateTime), cancelAsks.Price, seriestype = :scatter, marker = (:red, stroke(:red), :xcross, 0.7), label = "Cancel Ask (LO)")
    plot!(bubblePlot, Time.(cancelBids.DateTime), cancelBids.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), :xcross, 0.7), label = "Cancel Bid (LO)")
    plot!(bubblePlot, Time.(l1lob.DateTime), l1lob.MidPrice, seriestype = :line, linecolor = :black, label = "Mid-price")
    plot!(bubblePlot, Time.(l1lob.DateTime), l1lob.MicroPrice, seriestype = :line, linecolor = :green, label = "Micro-price")
    savefig(bubblePlot, "Figures/BubblePlot.pdf")
end
#---------------------------------------------------------------------------------------------------
