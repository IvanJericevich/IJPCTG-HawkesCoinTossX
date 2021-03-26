#=
HawkesImplementation:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Dieter Hendricks, Tim Gebbie
- Function: Submit 10-variate Hawkes simulation to CoinTossX
- Event types:
    1. MO to buy
    2. MO to sell
    3. Aggressive LO to buy
    4. Aggressive LO to sell
    5. Passive LO to buy
    6. Passive LO to sell
    7. Aggressive cancellation to buy
    8. Aggressive cancellation to sell
    9. Passive cancellation to buy
    10. Passive cancellation to sell
- Notes:
    Main thing to check here is that our parameters result in a relatively balanced system where the number of events reducing the liquidity is roughly the same as the
    number of events increasing the liquidity i.e. ∑(3+4+5+6) ≈ ∑(1+2+7+8+9+10). The event counts of these should be roughly the same.
=#
using DataFrames, Dates
cd(@__DIR__); clearconsole()
include("Hawkes.jl")
include("CoinTossXUtilities.jl")
function RPareto(xn, α, n = 1)
    return xn ./ (rand(n) .^ (1 / α))
end
#---------------------------------------------------------------------------------------------------

#----- Model 1 -----#
function InjectSimulation(arrivals; seed = 1)
    Random.seed!(seed)
    StartJVM()
    client = Login(1, 1)
    try # This ensures that the client gets logged out whether an error occurs or not
        SubmitOrder(client, Order("1", "Buy", "Limit", RPareto(20, 1)[1], 45))
        SubmitOrder(client, Order("2", "Buy", "Limit", RPareto(20, 1)[1], 45))
        SubmitOrder(client, Order("3", "Buy", "Limit", RPareto(20, 1)[1], 44))
        SubmitOrder(client, Order("4", "Sell", "Limit", RPareto(20, 1)[1], 50))
        SubmitOrder(client, Order("5", "Sell", "Limit", RPareto(20, 1)[1], 50))
        SubmitOrder(client, Order("6", "Sell", "Limit", RPareto(20, 1)[1], 51))
        SubmitOrder(client, Order(arrivals.OrderId[1], arrivals.Side[1], "Limit", arrivals.Volume[1], 44))
        arrivals.arrivalTime = arrivals.DateTime .+ Time(now())
        Juno.progress() do id # Progress bar
            for i in 2:nrow(arrivals)
                bestBid = ReceiveMarketData(client, :Bid, :Price); bestAsk = ReceiveMarketData(client, :Ask, :Price)
                if bestBid == 0 || bestAsk == 0 # If both sides are empty quit simulation
                    error("Both or one of the sides of the LOB have emptied")
                end
                if arrivals.Type[i] == :LO # Limit order
                    limitOrder = arrivals[i, :]
                    if limitOrder.Side == "Buy"
                        price = limitOrder.Passive ? bestBid - 1 : bestBid + 1
                    else
                        price = limitOrder.Passive ? bestAsk + 1 : bestAsk - 1
                    end
                    limitOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - limitOrder.arrivalTime)) : sleep(limitOrder.arrivalTime - Time(now()))
                    SubmitOrder(client, Order(limitOrder.OrderId, limitOrder.Side, "Limit", limitOrder.Volume, price))
                elseif arrivals.Type[i] == :MO # Market order
                    marketOrder = arrivals[i, :]
                    marketOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - marketOrder.arrivalTime)) : sleep(marketOrder.arrivalTime - Time(now()))
                    SubmitOrder(client, Order(marketOrder.OrderId, marketOrder.Side, "Market", marketOrder.Volume))
                else # Order cancel
                    cancelOrder = arrivals[i, :]
                    (LOBSnapshot, best) = cancelOrder.Side == "Buy" ? (ReceiveLOBSnapshot(client, "Buy"), ReceiveMarketData(client, :Bid, :Price)) : (ReceiveLOBSnapshot(client, "Sell"), ReceiveMarketData(client, :Ask, :Price))
                    OrderIds = cancelOrder.Passive ? [k for (k,v) in LOBSnapshot if v.Price != best] : [k for (k,v) in LOBSnapshot if v.Price == best]
                    if !isempty(OrderIds)
                        orderId = rand(OrderIds) # Passive => sample from orders not in L1; aggressive => sample from orders in L1
                        price = LOBSnapshot[orderId].Price # Get the price of the corresponding orderId
                        cancelOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - cancelOrder.arrivalTime)) : sleep(cancelOrder.arrivalTime - Time(now()))
                        CancelOrder(client, orderId, cancelOrder.Side, price)
                    end
                end
                @info "Trading" progress=(arrivals.DateTime[i] / arrivals.DateTime[end]) _id=id # Update progress
            end
        end
    finally
        Logout(client)
    end
end
#---------------------------------------------------------------------------------------------------

#----- Model 2 -----#
function InjectSimulation(arrivals; seed = 1)
    Random.seed!(seed)
    StartJVM()
    client = Login(1, 1)
    try # This ensures that the client gets logged out whether an error occurs or not
        SubmitOrder(client, Order("1", "Buy", "Limit", RPareto(20, 1)[1], 45))
        SubmitOrder(client, Order("2", "Buy", "Limit", RPareto(20, 1)[1], 45))
        SubmitOrder(client, Order("3", "Buy", "Limit", RPareto(20, 1)[1], 44))
        SubmitOrder(client, Order("4", "Sell", "Limit", RPareto(20, 1)[1], 50))
        SubmitOrder(client, Order("5", "Sell", "Limit", RPareto(20, 1)[1], 50))
        SubmitOrder(client, Order("6", "Sell", "Limit", RPareto(20, 1)[1], 51))
        SubmitOrder(client, Order(arrivals.OrderId[1], arrivals.Side[1], "Limit", arrivals.Volume[1], 44))
        arrivals.arrivalTime = arrivals.DateTime .+ Time(now())
        Juno.progress() do id # Progress bar
            for i in 2:nrow(arrivals)
                bestBid = ReceiveMarketData(client, :Bid, :Price); bestAsk = ReceiveMarketData(client, :Ask, :Price)
                if bestBid == 0 && bestAsk == 0 # If both sides are empty quit simulation
                    error("Both sides of the LOB have emptied")
                end
                if arrivals.Type[i] == :LO # Limit order
                    limitOrder = arrivals[i, :]
                    price = SetLimitPrice(limitOrder, bestBid, bestAsk)
                    limitOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - limitOrder.arrivalTime)) : sleep(limitOrder.arrivalTime - Time(now()))
                    SubmitOrder(client, Order(arrivals.OrderId[i], arrivals.Side[i], "Limit", arrivals.Volume[i], price))
                elseif arrivals.Type[i] == :MO # Market order
                    marketOrder = arrivals[i, :]
                    marketOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - marketOrder.arrivalTime)) : sleep(marketOrder.arrivalTime - Time(now()))
                    SubmitOrder(client, Order(marketOrder.OrderId, marketOrder.Side, "Market", marketOrder.Volume))
                else # Order cancel
                    cancelOrder = arrivals[i, :]
                    (LOBSnapshot, best) = arrivals.Side[i] == "Buy" ? (ReceiveLOBSnapshot(client, "Buy"), ReceiveMarketData(client, :Bid, :Price)) : (ReceiveLOBSnapshot(client, "Sell"), ReceiveMarketData(client, :Ask, :Price))
                    OrderIds = cancelOrder.Passive ? [k for (k,v) in LOBSnapshot if v.Price != best] : [k for (k,v) in LOBSnapshot if v.Price == best]
                    if !isempty(OrderIds)
                        orderId = rand(OrderIds) # Passive => sample from orders not in L1; aggressive => sample from orders in L1
                        price = LOBSnapshot[orderId].Price # Get the price of the corresponding orderId
                        cancelOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - cancelOrder.arrivalTime)) : sleep(cancelOrder.arrivalTime - Time(now()))
                        CancelOrder(client, orderId, arrivals.Side[i], price)
                    end
                end
                @info "Trading" progress=(arrivals.DateTime[i] / arrivals.DateTime[end]) _id=id # Update progress
            end
        end
    finally
        Logout(client)
    end
end
function SetLimitPrice(limitOrder, bestBid, bestAsk)
    if limitOrder.Side == "Buy" # Buy LO
        if limitOrder.Passive # Passive buy LO
            if bestAsk != 0 # If the ask side LOB is non-empty
                price = bestBid != 0 ? bestBid - 1 : bestAsk - 5 # If the bid side is non empty place the price 1 tick below the best; otherwise place it 5 ticks below the best ask (since the best bid is zero)
            else # If the ask side is empty (implies the bid side cannot be empty since an error would have thrown)
                price = bestBid - 1 # Place price 1 tick below best
            end
        else # Aggressive buy LO
            if bestBid != 0 # If the bid side LOB is non-empty
                spread = abs(bestAsk - bestBid)
                price = spread > 1 ? bestBid + 1 : bestBid # If the spead is greater than 1 tick place price above best; otherwise place price at best (note that this still applies if bestAsk = 0)
            else # If the bid side is empty (implies the ask side cannot be empty)
                price = bestAsk - 5 # Place price 5 ticks below best ask
            end
        end
    else # Sell LO
        if limitOrder.Passive # Passive sell LO
            if bestBid != 0 # If the bid side LOB is non-empty
                price = bestAsk != 0 ? bestAsk + 1 : bestBid + 5 # If the ask side is non empty place the price 1 tick above the best; otherwise place it 5 ticks above the best bid (since the best ask is zero)
            else # If the bid side is empty (implies the ask side cannot be empty since an error would have thrown)
                price = bestAsk + 1 # Place price 1 tick above best
            end
        else # Aggressive sell LO
            if bestAsk != 0 # If the ask side LOB is non-empty
                spread = abs(bestAsk - bestBid)
                price = spread > 1 ? bestAsk - 1 : bestAsk # If the spead is greater than 1 tick place price above best; otherwise place price at best (note that this still applies if bestBid = 0)
            else # If the ask side is empty (implies the ask side cannot be empty)
                price = bestBid + 5 # Place price 5 ticks abive best bid
            end
        end
    end
    return price
end
#---------------------------------------------------------------------------------------------------

#----- Implementation -----#
Random.seed!(5)
λ₀ = [0.01; 0.01; 0.02; 0.02; 0.02; 0.02; 0.015; 0.015; 0.015; 0.015]
β  = fill(0.2, 10, 10)
α  = [repeat([0.01], 10)'; repeat([0.01], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)']
arrivals = simulateHawkes(λ₀, α, β, 500; seed = 5) |> x -> DataFrame(DateTime = map(t -> Millisecond(round(Int, t * 1000)), reduce(vcat, x)),Type = reduce(vcat, fill.([:MO, :MO, :LO, :LO, :LO, :LO, :OC, :OC, :OC, :OC], length.(x))), Side = reduce(vcat, fill.(["Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell"], length.(x))),
OrderId = vcat(string.(6 .+ collect(1:sum(length.(x[1:6])))), fill("0", sum(length.(x[7:10])))), Volume = round.(Int, vcat(RPareto(20, 1.5, sum(length.(x[1:2]))), RPareto(20, 1, sum(length.(x[3:6]))), fill(0, sum(length.(x[7:10]))))),
Passive = reduce(vcat, fill.([false, false, false, false, true, true, false, false, true, true], length.(x))))
sort!(arrivals, :DateTime)
arrivals.DateTime .-= arrivals.DateTime[1]
delete!(arrivals, [1, 2])
InjectSimulation(arrivals, seed = 5)
#---------------------------------------------------------------------------------------------------
#=
reduceLiquid = 0; increaseLiquid = 0
redu = [3;4;5;6]; inc = [1;2;7;8;9;10]

for i in 1:length(redu)
    reduceLiquid += length(sim[redu[i]])
end
for i in 1:length(inc)
    increaseLiquid += length(sim[inc[i]])
end

reduceLiquid
increaseLiquid

function InjectSimulation(arrivals; seed = 1)
    Random.seed!(seed)
    StartJVM()
    client = Login(1, 1)
    skipCount = 0
    try # This ensures that the client gets logged out whether an error occurs or not

        SubmitOrder(client, Order("1", "Buy", "Limit", 100, 45))
        SubmitOrder(client, Order("2", "Buy", "Limit", 100, 45))
        SubmitOrder(client, Order("3", "Buy", "Limit", 100, 44))
        SubmitOrder(client, Order("4", "Buy", "Limit", 100, 44))
        SubmitOrder(client, Order("5", "Sell", "Limit", 100, 50))
        SubmitOrder(client, Order("6", "Sell", "Limit", 100, 50))
        SubmitOrder(client, Order("7", "Sell", "Limit", 100, 51))
        SubmitOrder(client, Order("8", "Sell", "Limit", 100, 51))

        SubmitOrder(client, Order(arrivals.OrderId[1], arrivals.Side[1], "Limit", arrivals.Volume[1], 50))
        arrivals.arrivalTime = arrivals.DateTime .+ Time(now())
        Juno.progress() do id # Progress bar
            for i in 2:nrow(arrivals)
                if arrivals.Type[i] == :LO # Limit order
                    bestBid = ReceiveMarketData(client, :Bid, :Price); bestAsk = ReceiveMarketData(client, :Ask, :Price)
                    if bestBid == 0 && bestAsk == 0 # If both sides are empty quit simulation
                        error("Both sides of the LOB have emptied")
                    end
                    spread = abs(bestAsk - bestBid)
                    if arrivals.Side[i] == "Buy"
                        if arrivals.Passive[i] # Place order 1 tick below best
                            price = bestBid != 0 ? bestBid - 1 : (skipCount += 1; continue) # If the ask side LOB is empty skip everything below (since sending an order would change the best)
                        else # If the spread is more than one tick improve the best, otherwise place at best
                            price = spread > 1 ? bestBid + 1 : bestBid # bestBid == 0 => price = 1; bestAsk == 0 => price = bestBid + 1; bestBid == 0 && bestAsk == 0 => error and finish
                        end
                    else
                        if arrivals.Passive[i] # Place order 1 tick above best
                            price = bestAsk != 0 ? bestAsk + 1 : (skipCount += 1; continue) # If the ask side LOB is empty skip everything below (since sending an order would change the best)
                        else # If the spread is more than one tick improve the best, otherwise place at best
                            price = spread > 1 ? bestAsk - 1 : bestAsk # bestBid == 0 => price = bestAsk - 1; bestAsk == 0 => price = 1
                        end
                    end
                    arrivals.arrivalTime[i] <= Time(now()) ? println(string("Timeout: ", Time(now()) - arrivals.arrivalTime[i])) : sleep(arrivals.arrivalTime[i] - Time(now()))
                    SubmitOrder(client, Order(arrivals.OrderId[i], arrivals.Side[i], "Limit", arrivals.Volume[i], price))
                elseif arrivals.Type[i] == :MO # Market order
                    best = arrivals.Side[i] == "Buy" ? ReceiveMarketData(client, :Ask, :Price) : ReceiveMarketData(client, :Bid, :Price) # Get the best on the contra side
                    if best != 0 # If the LOB is empty on the contra side do nothing
                        arrivals.arrivalTime[i] <= Time(now()) ? println(string("Timeout: ", Time(now()) - arrivals.arrivalTime[i])) : sleep(arrivals.arrivalTime[i] - Time(now()))
                        SubmitOrder(client, Order(arrivals.OrderId[i], arrivals.Side[i], "Market", arrivals.Volume[i]))
                    else
                        skipCount += 1
                    end
                else # Order cancel
                    (LOBSnapshot, best) = arrivals.Side[i] == "Buy" ? (ReceiveLOBSnapshot(client), ReceiveMarketData(client, :Bid)) : (ReceiveLOBSnapshot(client), ReceiveMarketData(client, :Ask))
                    if length(LOBSnapshot) > 1 # Atleast two orders in the LOB side
                        OrderIds = arrivals.Passive[i] ? [k for (k,v) in LOBSnapshot if (v.Side == arrivals.Side[i] && v.Price != best.Price)] : [k for (k,v) in LOBSnapshot if (v.Side == arrivals.Side[i] && v.Price == best.Price)]
                    elseif length(LOBSnapshot) == 1 && !arrivals.Passive[i] # Only order in the LOB is the best => only aggressive cancel is applicable
                        OrderIds = [k for (k,v) in LOBSnapshot if v == best]
                        println("Hello")
                    else # LOB is empty or passive cancel with only one order in the LOB => wait for next LO. Skip everything below
                        skipCount += 1; continue
                    end
                    if isempty(OrderIds)
                        skipCount += 1; continue
                    end
                    orderId = rand(OrderIds) # Passive => sample from orders not in L1; aggressive => sample from orders in L1
                    price = LOBSnapshot[orderId].Price # Get the price of the corresponding orderId
                    arrivals.arrivalTime[i] <= Time(now()) ? println(string("Timeout: ", Time(now()) - arrivals.arrivalTime[i])) : sleep(arrivals.arrivalTime[i] - Time(now()))
                    CancelOrder(client, orderId, arrivals.Side[i], price)
                end
                @info "Trading" progress=(arrivals.DateTime[i] / arrivals.DateTime[end]) _id=id # Update progress
            end
        end
    finally
        Logout(client)
        return skipCount
    end
end
=#
