#=
DataCleaning:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Clean CoinTossX simulated data into L1LOB and OHLCV data for stylized fact analysis
- Structure:
    1. Data preparation
    2. Supplementary functions
    3. Clean raw data into L1LOB and OHLCV data
- TODO: Check that cancel Ids are present in the LOB
=#
using CSV, DataFrames, Dates, Plots
clearconsole()
#---------------------------------------------------------------------------------------------------

#----- Data preparation -----#
rawOrders = CSV.File("Data/OrdersSubmitted_2.csv", drop = [:SecurityId, :OrderId], types = Dict(:ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64, :Side => Symbol), dateformat = "yyyy-mm-dd HH:MM:SS.s") |> DataFrame |> x -> filter(y -> y.Price != 0, x)
limitOrders = filter(x -> x.ClientOrderId > 0, rawOrders); cancelOrders = filter(x -> x.ClientOrderId < 0, rawOrders); marketOrders = CSV.File("Data/Trades_2.csv", drop = [:OrderId], types = Dict(:ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64), dateformat = "yyyy-mm-dd HH:MM:SS.s") |> DataFrame
cancelOrders.ClientOrderId .*= -1; cancelOrders[:, [:Price, :Volume]] = limitOrders[indexin(cancelOrders.ClientOrderId, limitOrders.ClientOrderId), [:Price, :Volume]]
marketOrders.Side = limitOrders.Side[indexin(marketOrders.ClientOrderId, limitOrders.ClientOrderId)] # Extract MO contra side
limitOrders.Type = fill(:LO, nrow(limitOrders)); cancelOrders.Type = fill(:OC, nrow(cancelOrders)); marketOrders.Type = fill(:MO, nrow(marketOrders))
orders = outerjoin(limitOrders, marketOrders, on = [:ClientOrderId, :DateTime, :Price, :Volume, :Type, :Side])
rename!(orders, :ClientOrderId => :OrderId)
sort!(orders, :DateTime)
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
    open("Data/Model1L1LOB.csv", "w") do file
        println(file, "DateTime,Price,Volume,Type,Side,MidPrice,MicroPrice,Spread")
        bids = Dict{Int64, Tuple{Int64, Int64}}(); asks = Dict{Int64, Tuple{Int64, Int64}}() # Both sides of the entire LOB are tracked with keys corresponding to orderIds
        bestBid = NamedTuple(); bestAsk = NamedTuple() # Current best bid/ask is stored in a tuple (Price, vector of Volumes, vector of OrderIds) and tracked
        animation = visualise ? Animation() : nothing
        Juno.progress() do id
            count = 1
            for line in eachrow(orders) # Iterate through all orders
                #-- Limit Orders --#
                if line.Type == :LO
                    if line.Side == :Buy # Buy limit order
                        if isempty(bids) || isempty(bestBid) # If the dictionary is empty, this order automatically becomes best
                            bestBid = (Price = line.Price, Volume = line.Volume, OrderId = [line.OrderId]) # New best is created
                            midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                            println(file, string(line.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",LO,1,", midPrice, ",", microPrice, ",", spread))
                        else # Otherwise find the best
                            if line.Price > bestBid.Price # Change best if price of current order better than the best
                                if !isempty(bestAsk) && line.Price >= bestAsk.Price # Bid crosses best ask => limit order becomes effective market order (to avoid an error first check if the otherside isn't empty)
                                    allowCrossing ? (count += 1; continue) : error("Negative spread - bid and ask have crossed at line " * string(count)) # Either disallow crossing or treat crossed order as effective market order (in which case ignore the limit order and only process subsequent market order)
                                end
                                bestBid = (Price = line.Price, Volume = line.Volume, OrderId = [line.OrderId]) # New best is created
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(line.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",LO,1,", midPrice, ",", microPrice, ",", spread))
                            elseif line.Price == bestBid.Price # Add the new order's volume and orderid to the best if they have the same price
                                bestBid = (Price = bestBid.Price, Volume = bestBid.Volume + line.Volume, OrderId = vcat(bestBid.OrderId, line.OrderId)) # Best is ammended by adding volume to best and appending the order id
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(line.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",LO,1,", midPrice, ",", microPrice, ",", spread))
                            end # Otherwise the L1LOB hasn't changed so do nothing
                        end
                        push!(bids, line.OrderId => (line.Price, line.Volume)) # New order is pushed to LOB dictionary only after best is processed
                    else # Sell limit order
                        if isempty(asks) || isempty(bestAsk) # If the dictionary is empty, this order automatically becomes best
                            bestAsk = (Price = line.Price, Volume = line.Volume, OrderId = [line.OrderId]) # New best is created
                            midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                            println(file, string(line.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",LO,-1,", midPrice, ",", microPrice, ",", spread))
                        else # Otherwise find the best
                            if line.Price < bestAsk.Price # Change best if price of current order better than the best
                                if !isempty(bestBid) && line.Price <= bestBid.Price # Ask crosses best bid => limit order becomes effective market order (to avoid an error first check if the otherside isn't empty)
                                    allowCrossing ? (count += 1; continue) : error("Negative spread - bid and ask have crossed at line " * string(count)) # Either disallow crossing or treat crossed order as effective market order (in which case ignore the limit order and only process subsequent market order since trade is printed in Trades.csv and will be handled anyway)
                                end
                                bestAsk = (Price = line.Price, Volume = line.Volume, OrderId = [line.OrderId]) # New best is created
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(line.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",LO,-1,", midPrice, ",", microPrice, ",", spread))
                            elseif line.Price == bestAsk.Price # Add the new order's volume and orderid to the best if they have the same price
                                bestAsk = (Price = bestAsk.Price, Volume = bestAsk.Volume + line.Volume, OrderId = vcat(bestAsk.OrderId, line.OrderId)) # Best is ammended by adding volume to best and appending the order id
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(line.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",LO,-1,", midPrice, ",", microPrice, ",", spread))
                            end # Otherwise the L1LOB hasn't changed so do nothing
                        end
                        push!(asks, line.OrderId => (line.Price, line.Volume)) # New order is pushed to LOB dictionary only after best is processed
                    end
                #-- Market Orders --#
                elseif line.Type == :MO  # Market order always affects the best
                    if line.Side == :Buy # Trade was buyer-initiated (Sell MO)
                        contraOrder = bids[line.OrderId]
                        # contraOrder = haskey(bids, line.OrderId) ? bids[line.OrderId] : bids[maximum(keys(bids))] # REVIEW: If the OrderId references itself then a Null Pointer will occur, so set the contraOrder to the price-time priority best (clientOrderId is in the same order as time)
                        println(file, string(line.DateTime, ",", line.Price, ",", line.Volume, ",MO,-1,missing,missing,missing")) # Sell trade is printed
                        if line.Volume == bestBid.Volume # Trade filled best - remove from LOB, and update best
                            delete!(bids, line.OrderId) # Remove the order from the LOB
                            if !isempty(bids) # If the LOB is non empty find the best
                                bestPrice = maximum(first.(collect(values(bids)))) # Find the new best price
                                indeces = findall(x -> first(x) == bestPrice, bids) # Find the order ids of the best
                                bestBid = (Price = bestPrice, Volume = sum(last(bids[i]) for i in indeces), OrderId = indeces) # Update the best
                            else # If the LOB is empty remove best
                                bestBid = NamedTuple()
                            end
                        else # Trade partially filled best
                            if line.Volume == contraOrder[2] # Trade filled contra order - remove order from LOB, remove order from best, and update best
                                delete!(bids, line.OrderId)
                                bestBid = (Price = bestBid.Price, Volume = bestBid.Volume - line.Volume, OrderId = setdiff(bestBid.OrderId, line.OrderId))
                            else # Trade partially filled contra order - update LOB, update best
                                bids[line.OrderId] = (contraOrder[1], contraOrder[2] - line.Volume)
                                bestBid = (Price = bestBid.Price, Volume = bestBid.Volume - line.Volume, OrderId = bestBid.OrderId)
                            end
                        end
                        midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                        !isempty(bestBid) ? println(file, string(line.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",LO,1,", midPrice, ",", microPrice, ",", spread)) : println(file, string(line.DateTime, ",missing,missing,LO,1,missing,missing,missing"))
                    else # Trade was seller-initiated (Buy MO)
                        contraOrder = asks[line.OrderId]
                        # contraOrder = haskey(asks, line.OrderId) ? asks[line.OrderId] : asks[maximum(keys(asks))] # REVIEW: If the OrderId references itself then a Null Pointer will occur, so set the contraOrder to the price-time priority best (clientOrderId is in the same order as time)
                        println(file, string(line.DateTime, ",", line.Price, ",", line.Volume, ",MO,1,missing,missing,missing")) # Buy trade is printed
                        if line.Volume == bestAsk.Volume # Trade filled best - remove from LOB, and update best
                            delete!(asks, line.OrderId) # Remove the order from the LOB
                            if !isempty(asks) # If the LOB is non empty find the best
                                bestPrice = maximum(first.(collect(values(asks)))) # Find the new best price
                                indeces = findall(x -> first(x) == bestPrice, asks) # Find the order ids of the best
                                bestAsk = (Price = bestPrice, Volume = sum(last(asks[i]) for i in indeces), OrderId = indeces) # Update the best
                            else # If the LOB is empty remove best
                                bestAsk = NamedTuple()
                            end
                        else # Trade partially filled best
                            if line.Volume == contraOrder[2] # Trade filled contra order - remove order from LOB, remove order from best, and update best
                                delete!(asks, line.OrderId)
                                bestAsk = (Price = bestAsk.Price, Volume = bestAsk.Volume - line.Volume, OrderId = setdiff(bestAsk.OrderId, line.OrderId))
                            else # Trade partially filled contra order - update LOB, update best
                                asks[line.OrderId] = (contraOrder[1], contraOrder[2] - line.Volume)
                                bestAsk = (Price = bestAsk.Price, Volume = bestAsk.Volume - line.Volume, OrderId = bestAsk.OrderId)
                            end
                        end
                        midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                        !isempty(bestAsk) ? println(file, string(line.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",LO,-1,", midPrice, ",", microPrice, ",", spread)) : println(file, string(line.DateTime, ",missing,missing,LO,-1,missing,missing,missing"))
                    end
                #-- Cancel Orders --#
                else
                    if line.Side == :Buy # Cancel buy limit order
                        delete!(bids, line.OrderId) # Remove the order from the LOB
                        if line.OrderId in bestBid.OrderId # Cancel hit the best
                            activeL1Orders = setdiff(bestBid.OrderId, line.OrderId) # Find the remaining level 1 order ids
                            if !isempty(activeL1Orders) # Cancel did not empty the best - remove order from best and update best
                                bestBid = (Price = bestBid.Price, Volume = bestBid.Volume - line.Volume, OrderId = activeL1Orders)
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(line.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",OC,1,", midPrice, ",", microPrice, ",", spread))
                            else # Cancel emptied the best
                                if !isempty(bids) # Orders still remain in the buy side LOB - update best
                                    bestPrice = maximum(first.(collect(values(bids)))) # Find the new best price
                                    indeces = findall(x -> first(x) == bestPrice, bids) # Find the order ids of the best
                                    bestBid = (Price = bestPrice, Volume = sum(last(bids[i]) for i in indeces), OrderId = indeces) # Update the best
                                    midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                    println(file, string(line.DateTime, ",", bestBid.Price, ",", bestBid.Volume, ",OC,1,", midPrice, ",", microPrice, ",", spread))
                                else # The buy side LOB was emptied - update best
                                    bestBid = NamedTuple()
                                    println(file, string(line.DateTime, ",missing,missing,OC,1,missing,missing,missing"))
                                end
                            end
                        end
                    else # Cancel sell limit order
                        delete!(asks, line.OrderId) # Remove the order from the LOB
                        if line.OrderId in bestAsk.OrderId # Cancel hit the best
                            activeL1Orders = setdiff(bestAsk.OrderId, line.OrderId) # Find the remaining level 1 order ids
                            if !isempty(activeL1Orders) # Cancel did not empty the best - remove order from best and update best
                                bestAsk = (Price = bestAsk.Price, Volume = bestAsk.Volume - line.Volume, OrderId = activeL1Orders)
                                midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                println(file, string(line.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",OC,-1,", midPrice, ",", microPrice, ",", spread))
                            else # Cancel emptied the best
                                if !isempty(asks) # Orders still remain in the sell side LOB - update best
                                    bestPrice = maximum(first.(collect(values(asks)))) # Find the new best price
                                    indeces = findall(x -> first(x) == bestPrice, asks) # Find the order ids of the best
                                    bestAsk = (Price = bestPrice, Volume = sum(last(asks[i]) for i in indeces), OrderId = indeces) # Update the best
                                    midPrice = MidPrice(bestBid, bestAsk); microPrice = MicroPrice(bestBid, bestAsk); spread = Spread(bestBid, bestAsk)
                                    println(file, string(line.DateTime, ",", bestAsk.Price, ",", bestAsk.Volume, ",OC,-1,", midPrice, ",", microPrice, ",", spread))
                                else # The sell side LOB was emptied - update best
                                    bestAsk = NamedTuple()
                                    println(file, string(line.DateTime, ",missing,missing,OC,-1,missing,missing,missing"))
                                end
                            end
                        end
                    end
                end
                if visualise # Visualisation
                    PlotLOBSnapshot(bids, asks)
                    frame(animation)
                end
                count += 1
                @info "Cleaning:" progress=(count / nrow(orders)) _id=id # Update progress
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
        barTimes = orders.DateTime[1]):resolution:orders.DateTime[end]
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
CleanData(orders, allowCrossing = true)
