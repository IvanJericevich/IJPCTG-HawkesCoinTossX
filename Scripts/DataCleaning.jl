#=
DataCleaning:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Clean CoinTossX simulated data into L1LOB and OHLCV data for stylized fact analysis as well as visualisation of HFT time-series
- Structure:
    1. Data preparation
    2. Supplementary functions
    3. Clean raw data into L1LOB format
    4. Plot simulation results
    5. Extract OHLCV data
- Examples:
    taqData = PrepareData("Model1/OrdersSubmitted_1", "Model1/Trades_1") |> taq -> CleanData(taq; allowCrossing = true)
    hawkesData = PrepareHawkesData(taqData)
    VisualiseSimulation(taqData, "Model1/L1LOB"; format = "png")
=#
using CSV, DataFrames, Dates, Plots, Statistics
clearconsole()
#---------------------------------------------------------------------------------------------------

#----- Data preparation -----#
#=
Function:
    - Prepare raw data for cleaning
    - orders submitted file contains duplicates of un-split trades. These are assigned the "WalkingMO" order type. This is required for Hawkes event classification
Arguments:
    - ordersSubmitted = the name of the orders submitted file
    - trades = the name of the trades file
Output:
    - TAQ data
=#
function PrepareData(ordersSubmitted::String, trades::String)
    # Limit and cancel orders
    rawOrders = CSV.File(string("Data/", ordersSubmitted, ".csv"), drop = [:SecurityId, :OrderId], types = Dict(:ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64, :Side => Symbol), dateformat = "yyyy-mm-dd HH:MM:SS.s") |> DataFrame
    limitOrders = filter(x -> x.ClientOrderId > 0, rawOrders) # Limit orders have positive ID
    limitOrders.Type = fill(:LO, nrow(limitOrders))
    limitOrders.Type[findall(x -> x == 0, limitOrders.Price)] .= :WalkingMO # Trades are included in the OrdersSubmitted file with their full (aggregated) volumes
    cancelOrders = filter(x -> x.ClientOrderId < 0, rawOrders) # Cancel orders have negative ID
    cancelOrders.ClientOrderId .*= -1 # Make cancel IDs positive again
    cancelOrders.Type = fill(:OC, nrow(cancelOrders)) # Note that you cannot associate cancels with their LO volumes since its possible for a trade to partially fill the LO before it is cancelled
    # Market orders
    marketOrders = CSV.File(string("Data/", trades, ".csv"), drop = [:OrderId], types = Dict(:ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64), dateformat = "yyyy-mm-dd HH:MM:SS.s") |> DataFrame
    marketOrders.Side = limitOrders.Side[indexin(marketOrders.ClientOrderId, limitOrders.ClientOrderId)] # Extract MO contra side
    marketOrders.Type = fill(:MO, nrow(marketOrders))
    orders = outerjoin(limitOrders, cancelOrders, marketOrders, on = [:ClientOrderId, :DateTime, :Price, :Volume, :Type, :Side]) # Combine everything
    rename!(orders, :ClientOrderId => :OrderId)
    sort!(orders, :DateTime)
    return orders
end
function PrepareHawkesData(orders::DataFrame)
    previousTime = Millisecond(2)
    data = DataFrame(DateTime = Vector{Float64}(), Volume = Vector{Int64}(), Type = Vector{Symbol}(), Side = Vector{Symbol}(), IsAggressive = Vector{Bool}())
    for i in 1:nrow(orders)
        order = orders[i, :]
        if order.Type == :CLO # Crossing limit order => record the EMO first
            aggregateVolume = 0
            count = 1
            while orders.Type[i + count] == :EMO
                aggregateVolume += orders.Volume[i + count]
                count += 1
            end
            side = order.Side == :Buy ? :Sell : :Buy # MO side is opposite to contra side
            time = order.DateTime == previousTime ? order.DateTime + Millisecond(1) : order.DateTime
            push!(data, (Dates.value(time) / 1000, aggregateVolume, :WalkingMO, side, true)) # Push the aggregated EMO
            if aggregateVolume != order.Volume
                time += Millisecond(1)
                push!(data, (Dates.value(time) / 1000, order.Volume - aggregateVolume, :LO, order.Side, true)) # Push the crossed LO with DateTime modified by 1 millisecond (SHOULD I MINUS AGGREGATE VOLUME)
            end
            previousTime = time
        elseif order.Type != :EMO && order.Type != :MO # Disregard split MOs and EMOs since they are aggregated in the above case
            time = order.DateTime == previousTime ? order.DateTime + Millisecond(1) : order.DateTime
            push!(data, (Dates.value(time) / 1000, order.Volume, order.Type, order.Side, order.IsAggressive))
            previousTime = time
        end
    end
    events = [(:WalkingMO, :Buy, true), (:WalkingMO, :Sell, true), (:LO, :Buy, true), (:LO, :Sell, true), (:LO, :Buy, false), (:LO, :Sell, false), (:OC, :Buy, true), (:OC, :Sell, true), (:OC, :Buy, false), (:OC, :Sell, false)]
    hawkesData = groupby(data, [:Type, :Side, :IsAggressive]) |> x -> map(event -> collect(x[event].DateTime), events)
    return hawkesData
end
#---------------------------------------------------------------------------------------------------

#----- Supplementary functions -----#
function MidPrice(best::NamedTuple, contraBest::NamedTuple)::Union{Missing, Float64}
    return (isempty(best) || isempty(contraBest)) ? missing : (best.Price + contraBest.Price) / 2
end
function MicroPrice(best::NamedTuple, contraBest::NamedTuple)::Union{Missing, Float64}
    return (isempty(best) || isempty(contraBest)) ? missing : (best.Price * best.Volume + contraBest.Price * contraBest.Volume) / (best.Volume + contraBest.Volume)
end
function Spread(best::NamedTuple, contraBest::NamedTuple)::Union{Missing, Float64}
    return (isempty(best) || isempty(contraBest)) ? missing : abs(best.Price - contraBest.Price)
end
function OrderImbalance(bids::Dict{Int64, Tuple{Int64, Int64}}, asks::Dict{Int64, Tuple{Int64, Int64}})::Union{Missing, Float64}
    if isempty(bids) && isempty(asks)
        return missing
    elseif isempty(bids)
        return - sum(last.(collect(values(asks))))
    elseif isempty(asks)
        return sum(last.(collect(values(bids))))
    else
        totalBuyVolume = sum(last.(collect(values(bids))))
        totalSellVolume = sum(last.(collect(values(asks))))
        return totalBuyVolume - totalSellVolume
    end
end
#---------------------------------------------------------------------------------------------------

#----- Clean raw data into L1LOB format -----#
#=
Function:
    - Update LOB and best with LO
    - Full crossed orders are also added to the LOB and then aggressed with subsequent effective MOs
    - Classify LO event as either aggressive or passive
    - Append mid-price, micro-price and spread info
Arguments:
    - file = output file to which L1LOB will be printed
    - order = order to be processed
    - best = best bid (ask) if bid (ask) LO
    - contraBest = best ask (bid) if ask (bid) LO
    - lob = bid (ask) side of LOB if bid (ask) LO
    - isAggressive = vector of past classifications
    - side = ∈ {-1, 1}
    - allowCrossing = should crossed orders be handled or not
Output:
    - Best bid/ask
    - Order id of crossed order (if any)
=#
function ProcessLimitOrder!(file::IOStream, order::DataFrameRow, best::NamedTuple, contraBest::NamedTuple, lob::Dict{Int64, Tuple{Int64, Int64}}, isAggressive::Vector{Bool}, side::Int64, allowCrossing::Bool)
    crossedOrderId = nothing # Initialize. This resets when a new LO occurs
    if isempty(lob) || isempty(best) # If the dictionary is empty, this order automatically becomes best
        best = (Price = order.Price, Volume = order.Volume, OrderId = [order.OrderId]) # New best is created
        midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
        println(file, string(order.DateTime, ",", best.Price, ",", best.Volume, ",LO,", side, ",", midPrice, ",", microPrice, ",", spread))
        push!(isAggressive, true)
    else # Otherwise find the best
        if (side * order.Price) > (side * best.Price) # Change best if price of current order better than the best (side == 1 => order.Price > best.Price) (side == -1 => order.Price < best.Price)
            best = (Price = order.Price, Volume = order.Volume, OrderId = [order.OrderId]) # New best is created. This includes crossed orders since they are handled later when the EMOs are processed
            if !isempty(contraBest) && (side * order.Price) >= (side * contraBest.Price) # Crossing order => limit order becomes effective market order (to avoid an error first check if the otherside isn't empty)
                if allowCrossing # Either disallow crossing or treat crossed order as effective market order at the next iteration
                    println(string("Order ", order.OrderId, " crossed the spread"))
                    crossedOrderId = order.OrderId # Update the L1LOB with the crossed order. The trades occuring hereafter will be deducted from this as well as from the best on the contra side
                    order.Type = :CLO # Change the order type
                else
                    error("Negative spread at order " * string(order.OrderId))
                end
            else # Only print the LO if it is not a CLO. The CLO is only printed if it is partially filled after aggressing against the contra side
                midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
                println(file, string(order.DateTime, ",", best.Price, ",", best.Volume, ",LO,", side, ",", midPrice, ",", microPrice, ",", spread))
            end
            push!(isAggressive, true)
        elseif order.Price == best.Price # Add the new order's volume and orderid to the best if they have the same price
            best = (Price = best.Price, Volume = best.Volume + order.Volume, OrderId = vcat(best.OrderId, order.OrderId)) # Best is ammended by adding volume to best and appending the order id
            midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
            println(file, string(order.DateTime, ",", best.Price, ",", best.Volume, ",LO,", side, ",", midPrice, ",", microPrice, ",", spread))
            push!(isAggressive, true)
        else # Otherwise the L1LOB hasn't changed so do nothing
            push!(isAggressive, false) # Order did not affect L1LOB
        end
    end
    push!(lob, order.OrderId => (order.Price, order.Volume)) # New order is always pushed to LOB dictionary only after best is processed
    return best, crossedOrderId
end
#=
Function:
    - Update LOB and best with MO
    - Append mid-price, micro-price and spread info
Arguments:
    - file = output file to which L1LOB will be printed
    - order = order to be processed
    - best = best bid (ask) if bid (ask) LO
    - contraBest = best ask (bid) if ask (bid) LO
    - lob = bid (ask) side of LOB if bid (ask) LO
    - isAggressive = vector of past classifications
    - side = ∈ {-1, 1}
Output:
    - Best bid/ask
=#
function ProcessMarketOrder!(file::IOStream, order::DataFrameRow, best::NamedTuple, contraBest::NamedTuple, lob::Dict{Int64, Tuple{Int64, Int64}}, side::Int64)
    contraOrder = lob[order.OrderId] # Extract order on contra side
    if order.Volume == best.Volume # Trade filled best - remove from LOB, and update best
        delete!(lob, order.OrderId) # Remove the order from the LOB
        if !isempty(lob) # If the LOB is non empty find the best
            bestPrice = side * maximum(side .* first.(collect(values(lob)))) # Find the new best price (bid => side == 1 so find max price) (ask => side == -1 so find min price)
            indeces = [k for (k,v) in lob if first(v) == bestPrice] # Find the order ids of the best
            best = (Price = bestPrice, Volume = sum(last(lob[i]) for i in indeces), OrderId = indeces) # Update the best
        else # If the LOB is empty remove best
            best = NamedTuple()
        end
    else # Trade partially filled best
        if order.Volume == contraOrder[2] # Trade filled contra order - remove order from LOB, remove order from best, and update best
            delete!(lob, order.OrderId)
            best = (Price = best.Price, Volume = best.Volume - order.Volume, OrderId = setdiff(best.OrderId, order.OrderId))
        else # Trade partially filled contra order - update LOB, update best
            lob[order.OrderId] = (contraOrder[1], contraOrder[2] - order.Volume)
            best = (Price = best.Price, Volume = best.Volume - order.Volume, OrderId = best.OrderId)
        end
    end
    midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
    !isempty(best) ? println(file, string(order.DateTime, ",", best.Price, ",", best.Volume, ",LO,", side, ",", midPrice, ",", microPrice, ",", spread)) : println(file, string(order.DateTime, ",missing,missing,LO,", side, ",missing,missing,missing"))
    return best
end
#=
Function:
    - Update LOB and best with OC
    - Classify OC event as either aggressive or passive
    - Append mid-price, micro-price and spread info
Arguments:
    - file = output file to which L1LOB will be printed
    - order = order to be processed
    - best = best bid (ask) if bid (ask) LO
    - contraBest = best ask (bid) if ask (bid) LO
    - lob = bid (ask) side of LOB if bid (ask) LO
    - isAggressive = vector of past classifications
    - side = ∈ {-1, 1}
Output:
    - Best bid/ask
=#
function ProcessCancelOrder!(file::IOStream, order::DataFrameRow, best::NamedTuple, contraBest::NamedTuple, lob::Dict{Int64, Tuple{Int64, Int64}}, isAggressive::Vector{Bool}, side::Int64)
    delete!(lob, order.OrderId) # Remove the order from the LOB
    if order.OrderId in best.OrderId # Cancel hit the best
        if !isempty(lob) # Orders still remain in the LOB - find and update best
            bestPrice = side * maximum(side .* first.(collect(values(lob)))) # Find the new best price (bid => side == 1 so find max price) (ask => side == -1 so find min price)
            indeces = [k for (k,v) in lob if first(v) == bestPrice] # Find the order ids of the best
            best = (Price = bestPrice, Volume = sum(last(lob[i]) for i in indeces), OrderId = indeces) # Update the best
            midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
            println(file, string(order.DateTime, ",", best.Price, ",", best.Volume, ",OC,", side, ",", midPrice, ",", microPrice, ",", spread))
        else # The buy side LOB was emptied - update best
            best = NamedTuple()
            println(file, string(order.DateTime, ",missing,missing,OC,", side, ",missing,missing,missing"))
        end
        push!(isAggressive, true)
    else # OC did not hit best
        push!(isAggressive, false)
    end
    return best
end
#=
Function:
    - Update LOB and best with effective MO. This requires us to aggress the order against both side of the LOB
    - Append mid-price, micro-price and spread info
Arguments:
    - file = output file to which L1LOB will be printed
    - order = order to be processed
    - nextOrder = the order type of the next order to check if a partially filled crossed LO needs to be printed
    - best = best bid (ask) if bid (ask) LO
    - contraBest = best ask (bid) if ask (bid) LO
    - lob = bid (ask) side of LOB if bid (ask) LO
    - isAggressive = vector of past classifications
    - side = ∈ {-1, 1}
Output:
    - Best bid/ask
    - The order id of the crossed order if it hasn't been fully handled. Otherwise nothing
=#
function ProcessEffectiveMarketOrder!(file::IOStream, order::DataFrameRow, nextOrder::Symbol, best::NamedTuple, contraBest::NamedTuple, lob::Dict{Int64, Tuple{Int64, Int64}}, crossedOrderId::Int64, side::Int64) # Remove the executed quantity from the LO and push the remaining volume to the LOB
    if order.Volume == best.Volume # Crossed LO (which is now at the best) was fully executed against contra side - remove from LOB, and update best. Note that all crossed orders will sit at the best
        delete!(lob, crossedOrderId) # Remove the order from the LOB
        if !isempty(lob) # If the LOB is non empty find the best
            bestPrice = side * maximum(side .* first.(collect(values(lob)))) # Find the new best price (bid => side == 1 so find max price) (ask => side == -1 so find min price)
            indeces = [k for (k,v) in lob if first(v) == bestPrice] # Find the order ids of the best
            best = (Price = bestPrice, Volume = sum(last(lob[i]) for i in indeces), OrderId = indeces) # Update the best
        else # If the LOB is empty remove best
            best = NamedTuple()
        end
        crossedOrderId = nothing # Crossed LO was filled so reset
    else # Trade partially filled best
        lob[crossedOrderId] = (lob[crossedOrderId][1], lob[crossedOrderId][2] - order.Volume) # Update the volume of the crossed LO in the LOB
        best = (Price = best.Price, Volume = best.Volume - order.Volume, OrderId = best.OrderId)
        if nextOrder != :MO # If the crossed LO has been partially filled but aggressed all contra best orders (no more EMOs left)
            midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
            println(file, string(order.DateTime, ",", best.Price, ",", best.Volume, ",CLO,", side, ",", midPrice, ",", microPrice, ",", spread)) # The best already updated so all thats needed is to print and set crossed order = nothing
            crossedOrderId = nothing # Reset the crossed order indicator as it has been fully handled (need not update crossed order indicator elsewhere)
        end
    end
    order.Type = :EMO # Change the order type
    return best, crossedOrderId # Update the crossed order indicator
end
#=
Function:
    - Process all orders and clean raw TAQ data into L1LOB bloomberg format
    - Classify events as either aggressive or passive
Arguments:
    - orders = TAQ data
    - visusalise = plot the lob through time
    - allowCrossing = should crossed orders be handled or not
Output:
    - TAQ data for Hawkes analysis
    - Output L1LOB file written to csv
=#
function CleanData(orders::DataFrame; visualise::Bool = false, allowCrossing::Bool = false)
    bids = Dict{Int64, Tuple{Int64, Int64}}(); asks = Dict{Int64, Tuple{Int64, Int64}}() # Both sides of the entire LOB are tracked with keys corresponding to orderIds
    bestBid = NamedTuple(); bestAsk = NamedTuple() # Current best bid/ask is stored in a tuple (Price, vector of Volumes, vector of OrderIds) and tracked
    crossedOrderId = nothing # Stores the order that crossed if it hasn't been fully handled yet. If it has been processed then contains nothing
    isAggressive = Vector{Bool}() # Classifies all events as either aggressive or passive
    imbalance = Vector{Union{Missing, Float64}}() # Calculate the order imbalance in the LOB
    animation = visualise ? Animation() : nothing
    open("Data/Model2/L1LOB.csv", "w") do file
        println(file, "DateTime,Price,Volume,Type,Side,MidPrice,MicroPrice,Spread") # Header
        Juno.progress() do id # Progress bar
            for i in 1:nrow(orders) # Iterate through all orders
                order = orders[i, :]
                #-- Limit Orders --#
                if order.Type == :LO
                    if order.Side == :Buy # Buy limit order
                        bestBid, crossedOrderId = ProcessLimitOrder!(file, order, bestBid, bestAsk, bids, isAggressive, 1, allowCrossing) # Add the order to the lob and update the best if necessary (crossed orders as well)
                    else # Sell limit order
                        bestAsk, crossedOrderId = ProcessLimitOrder!(file, order, bestAsk, bestBid, asks, isAggressive, -1, allowCrossing) # Add the order to the lob and update the best if necessary (crossed orders as well)
                    end
                #-- Market Orders --#
                elseif order.Type == :MO # Market order always affects the best
                    if order.Side == :Buy # Trade was buyer-initiated (Sell MO)
                        println(file, string(order.DateTime, ",", order.Price, ",", order.Volume, ",MO,-1,missing,missing,missing")) # Sell trade is always printed (EMOs are just classified as MOs)
                        bestBid = ProcessMarketOrder!(file, order, bestBid, bestAsk, bids, 1) # Sell trade affects bid side. Always aggress MO/EMO against contra side and update LOB and best. This is done before the processing of the crossed LO
                        if !isnothing(crossedOrderId) # The crossed order hasn't been fully handled yet. So after handling the above EMO aggress it against the crossed LO
                            bestAsk, crossedOrderId = ProcessEffectiveMarketOrder!(file, order, orders[i + 1, :Type], bestAsk, bestBid, asks, crossedOrderId, -1) # Aggress effective sell MO against ask side (originating crossed LO). Update crossed order at LOB and best. Print partially filled crossed LO if necessary
                        end
                    else # Trade was seller-initiated (Buy MO)
                        println(file, string(order.DateTime, ",", order.Price, ",", order.Volume, ",MO,1,missing,missing,missing")) # Buy trade is always printed (EMOs are just classified as MOs)
                        bestAsk = ProcessMarketOrder!(file, order, bestAsk, bestBid, asks, -1) # Buy trade affects ask side. Always aggress MO/EMO against contra side and update LOB and best. This is done before the processing of the crossed LO
                        if !isnothing(crossedOrderId) # The crossed order hasn't been fully handled yet. So after handling the above EMO aggress it against the crossed LO
                            bestBid, crossedOrderId = ProcessEffectiveMarketOrder!(file, order, orders[i + 1, :Type], bestBid, bestAsk, bids, crossedOrderId, 1) # Aggress effective buy MO against bid side (originating crossed LO). Update crossed order at  LOB and best. Print partially filled crossed LO if necessary
                        end
                    end
                    push!(isAggressive, true) # MO/EMO are always aggressive
                #-- Cancel Orders --#
                elseif order.Type == :OC
                    if order.Side == :Buy # Cancel buy limit order
                        bestBid = ProcessCancelOrder!(file, order, bestBid, bestAsk, bids, isAggressive, 1) # Aggress cancel order against buy side and update LOB and best
                    else # Cancel sell limit order
                        bestAsk = ProcessCancelOrder!(file, order, bestAsk, bestBid, asks, isAggressive, -1) # Aggress cancel order against sell side and update LOB and best
                    end
                #-- Un-split Market Orders --#
                else # WalkingMO is a trade duplicate which occurs in OrdersSubmitted and reflects the full amounts of trades (un-split trades)
                    push!(isAggressive, true) # Walking MO is always aggressive. Ignore these in cleaning since it is handled by split MOs
                end
                push!(imbalance, OrderImbalance(bids, asks)) # Calculate the volume imbalance after every iteration
                if visualise # Visualisation
                    PlotLOBSnapshot(bids, asks)
                    frame(animation)
                end
                @info "Cleaning:" progress=(i / nrow(orders)) _id=id # Update progress
            end
            visualise ? gif(animation, "LOB.gif", fps = 3) : nothing
        end
    end
    Juno.notification("Data cleaning complete"; kind = :Info, options = Dict(:dismissable => false))
    orders.IsAggressive = isAggressive # Append classification to data
    orders.DateTime = orders.DateTime .- orders.DateTime[1] # Reformat times into relative times (milliseconds)
    orders.Imbalance = imbalance # Append order imbalance to data
    return orders
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
#---------------------------------------------------------------------------------------------------

#----- Plot simulation results -----#
function VisualiseSimulation(orders::DataFrame, l1lob::String; format = "pdf", duration = missing)
    filter!(x -> x.Type != :WalkingMO, orders)
    orders.DateTime = Dates.value.(orders.DateTime) ./ 1000
    l1lob = CSV.File(string("Data/", l1lob, ".csv"), types = Dict(:Type => Symbol, :Spread => Float64), missingstring = "missing") |> DataFrame |> x -> filter(y -> x.Type != :MO && x.Type != :EMO, x) # Filter out trades from L1LOB since their mid-prices are missing
    l1lob.DateTime = Dates.value.(l1lob.DateTime .- l1lob.DateTime[1]) ./ 1000
    if !ismissing(duration)
        filter!(x -> x.DateTime <= duration, l1lob); filter!(x -> x.DateTime <= duration, orders)
    end
    asks = filter(x -> x.Type == :LO && x.Side == :Sell, orders); bids = filter(x -> x.Type == :LO && x.Side == :Buy, orders)
    sells = filter(x -> x.Type == :MO && x.Side == :Buy, orders); buys = filter(x -> x.Type == :MO && x.Side == :Sell, orders) # scale * (asks.Volume / mean(asks.Volume))
    cancelAsks = filter(x -> x.Type == :OC && x.Side == :Sell, orders); cancelBids = filter(x -> x.Type == :OC && x.Side == :Buy, orders)
    bubblePlot = plot(asks.DateTime, asks.Price, seriestype = :scatter, marker = (:red, stroke(:red), 0.7), label = "Ask (LO)", ylabel = "Price (ticks)", legend = :topleft, legendfontsize = 5, xrotation = 30)
    plot!(bubblePlot, bids.DateTime, bids.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), 0.7), label = "Bid (LO)")
    plot!(bubblePlot, sells.DateTime, sells.Price, seriestype = :scatter, marker = (:red, stroke(:red), :utriangle, 0.7), label = "Sell (MO)")
    plot!(bubblePlot, buys.DateTime, buys.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), :utriangle, 0.7), label = "Buy (MO)")
    plot!(bubblePlot, cancelAsks.DateTime, cancelAsks.Price, seriestype = :scatter, marker = (:red, stroke(:red), :xcross, 0.7), label = "Cancel Ask (LO)")
    plot!(bubblePlot, cancelBids.DateTime, cancelBids.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), :xcross, 0.7), label = "Cancel Bid (LO)")
    plot!(bubblePlot, l1lob.DateTime, l1lob.MidPrice, seriestype = :steppost, linecolor = :black, label = "Mid-price")
    plot!(bubblePlot, l1lob.DateTime, l1lob.MicroPrice, seriestype = :line, linecolor = :green, label = "Micro-price")
    volumeImbalance = plot(orders.DateTime, orders.Imbalance, seriestype = :line, linecolor = :purple, xlabel = "Time (s)", ylabel = "Order Imbalance (%)", label = "OrderImbalance", legend = :topleft, legendfontsize = 5)
    plot!(twinx(), l1lob.DateTime, l1lob.Spread, seriestype = :steppost, linecolor = :orange, xlabel = "Time (s)", ylabel = "Spread", label = "Spread", legend = :topright, legendfontsize = 5)
    l = @layout([a; b{0.3h}])
    simulation = plot(bubblePlot, volumeImbalance, layout = l, link = :x)
    savefig(simulation, "Figures/Simulation." * format)
end
#---------------------------------------------------------------------------------------------------
