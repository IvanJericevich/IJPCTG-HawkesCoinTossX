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
using DataFrames, Dates#, Optim#, CSV
clearconsole()
include(pwd() * "/Scripts/Hawkes.jl")
include(pwd() * "/Scripts/DataCleaning.jl")
include(pwd() * "/Scripts/CoinTossXUtilities.jl")
λ₀ = [0.01; 0.01; 0.02; 0.02; 0.02; 0.02; 0.015; 0.015; 0.015; 0.015]
α = reduce(hcat, fill(λ₀, 10)) # α  = [repeat([0.01], 10)'; repeat([0.01], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)']
β  = fill(0.2, 10, 10)
function RPareto(xn, α, n = 1)
    return xn ./ (rand(n) .^ (1 / α))
end
#---------------------------------------------------------------------------------------------------

#----- Simulation -----#
Random.seed!(5)
arrivals = ThinningSimulation(λ₀, α, β, 28800; seed = 5) |> x -> DataFrame(DateTime = map(t -> Millisecond(round(Int, t * 1000)), reduce(vcat, x)),Type = reduce(vcat, fill.([:MO, :MO, :LO, :LO, :LO, :LO, :OC, :OC, :OC, :OC], length.(x))), Side = reduce(vcat, fill.(["Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell"], length.(x))),
Volume = round.(Int, vcat(RPareto(50, 1, sum(length.(x[1:2]))), RPareto(20, 1, sum(length.(x[3:6]))), fill(0, sum(length.(x[7:10]))))), Passive = reduce(vcat, fill.([false, false, false, false, true, true, false, false, true, true], length.(x))))
sort!(arrivals, :DateTime)
delete!(arrivals, 1)
arrivals.OrderId = string.(collect(1:nrow(arrivals)))
arrivals.DateTime .-= arrivals.DateTime[1]
# arrivals.DateTime .÷= 2
InjectSimulation(arrivals, seed = 5)
#---------------------------------------------------------------------------------------------------
mo = sum(arrivals.Volume[findall(x -> x == :MO, arrivals.Type)])
lo = sum(arrivals.Volume[findall(x -> x == :LO, arrivals.Type)])
#----- Hawkes Recalibration -----#
events = [(:WalkingMO, :Buy, true), (:WalkingMO, :Sell, true), (:LO, :Buy, true), (:LO, :Sell, true), (:LO, :Buy, false), (:LO, :Sell, false), (:OC, :Buy, true), (:OC, :Sell, true), (:OC, :Buy, false), (:OC, :Sell, false)]
data = PrepareData("Model2/OrdersSubmitted_1", "Model2/Trades_1") |> x -> CleanData(x) |> y -> groupby(y, [:Type, :Side, :IsAggressive]) |> z -> map(event -> Dates.value.(collect(z[event].DateTime)) ./ 1000, events)
initialSolution = log.(vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1))))
logLikelihood = TwiceDifferentiable(θ -> Calibrate(exp.(θ), data, 28800, 10), initialSolution, autodiff = :forward)
@time calibratedParameters = optimize(logLikelihood, initialSolution, LBFGS(), Optim.Options(show_trace = true))
#=
open("Parameters.txt", "w") do file
    for p in exp.(Optim.minimizer(calibratedParameters))
        println(file, p)
    end
end
MAE = mean(abs.(exp.(Optim.minimizer(calibratedParameters)) - initialSolution))
=#
#---------------------------------------------------------------------------------------------------

#----- Model 1 -----#
function InjectSimulation(arrivals; seed = 1)
    StartJVM()
    client = Login(1, 1)
    try # This ensures that the client gets logged out whether an error occurs or not
        SubmitOrder(client, Order(arrivals.OrderId[1], arrivals.Side[1], "Limit", arrivals.Volume[1], 1000))
        arrivals.arrivalTime = arrivals.DateTime .+ Time(now())
        previousBestBid = previousBestAsk = 0
        Juno.progress() do id # Progress bar
            for i in 2:nrow(arrivals)
                bestBid = ReceiveMarketData(client, :Bid, :Price); bestAsk = ReceiveMarketData(client, :Ask, :Price)
                if arrivals.Type[i] == :LO # Limit order
                    limitOrder = arrivals[i, :]
                    if bestBid == 0 && bestAsk == 0 # If both sides are empty => implement fail safe
                        println("Both sides of the LOB emptied")
                        price = limitOrder.Side == "Buy" ? SetLimitPrice(limitOrder, bestBid, previousBestAsk, seed) : SetLimitPrice(limitOrder, previousBestBid, bestAsk, seed)
                    else
                        price = SetLimitPrice(limitOrder, bestBid, bestAsk, seed)
                    end
                    limitOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - limitOrder.arrivalTime)) : sleep(limitOrder.arrivalTime - Time(now()))
                    SubmitOrder(client, Order(limitOrder.OrderId, limitOrder.Side, "Limit", limitOrder.Volume, price))
                elseif arrivals.Type[i] == :MO # Market order
                    marketOrder = arrivals[i, :]
                    if (marketOrder.Side == "Buy" && bestAsk != 0) || (marketOrder.Side == "Sell" && bestBid != 0) # Don't submit a trade if the contra side is empty
                        marketOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - marketOrder.arrivalTime)) : sleep(marketOrder.arrivalTime - Time(now()))
                        SubmitOrder(client, Order(marketOrder.OrderId, marketOrder.Side, "Market", marketOrder.Volume))
                    end
                elseif arrivals.Type[i] == :OC # Order cancel
                    cancelOrder = arrivals[i, :]
                    if (cancelOrder.Side == "Buy" && bestBid != 0) || (cancelOrder.Side == "Sell" && bestAsk != 0) # Only send a cancel order through if the LOB is non-empty
                        (LOBSnapshot, best) = cancelOrder.Side == "Buy" ? (ReceiveLOBSnapshot(client, "Buy"), ReceiveMarketData(client, :Bid, :Price)) : (ReceiveLOBSnapshot(client, "Sell"), ReceiveMarketData(client, :Ask, :Price))
                        OrderIds = cancelOrder.Passive ? [k for (k,v) in LOBSnapshot if v.Price != best] : [k for (k,v) in LOBSnapshot if v.Price == best]
                        if !isempty(OrderIds)
                            Random.seed!(seed)
                            orderId = rand(OrderIds) # Passive => sample from orders not in L1; aggressive => sample from orders in L1
                            price = LOBSnapshot[orderId].Price # Get the price of the corresponding orderId
                            cancelOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - cancelOrder.arrivalTime)) : sleep(cancelOrder.arrivalTime - Time(now()))
                            CancelOrder(client, orderId, cancelOrder.Side, price)
                        end
                    end
                end
                if bestBid != 0 # Update previous best bid only if it is non-empty
                    previousBestBid = bestBid
                end
                if bestAsk != 0 # Update previous best ask only if it is non-empty
                    previousBestAsk = bestAsk
                end
                seed += 1 # Change seed for next iteration
                @info "Trading" progress=(arrivals.DateTime[i] / arrivals.DateTime[end]) _id=id # Update progress
            end
        end
    finally
        Logout(client)
    end
end
function SetLimitPrice(limitOrder, bestBid, bestAsk, seed)
    spreadFailSafe = collect(1:10) # Vector of ticks from which to sample randomly
    if limitOrder.Side == "Buy" # Buy LO
        if limitOrder.Passive # Passive buy LO
            if bestAsk != 0 # If the ask side LOB is non-empty
                Random.seed!(seed)
                price = bestBid != 0 ? bestBid - 1 : bestAsk - rand(spreadFailSafe) # If the bid side is non empty place the price 1 tick below the best; otherwise place it random number ticks below the best ask (since the best bid is zero)
            else # If the ask side is empty (implies the bid side cannot be empty since an error would have thrown)
                price = bestBid - 1 # Place price 1 tick below best
            end
        else # Aggressive buy LO
            if bestBid != 0 # If the bid side LOB is non-empty
                price = bestBid + 1 # Place price 1 tick above best bid even if spread = 1
            else # If the bid side is empty (implies the ask side cannot be empty)
                Random.seed!(seed)
                price = bestAsk - rand(spreadFailSafe) # Place price random number of ticks below best ask
            end
        end
    else # Sell LO
        if limitOrder.Passive # Passive sell LO
            if bestBid != 0 # If the bid side LOB is non-empty
                Random.seed!(seed)
                price = bestAsk != 0 ? bestAsk + 1 : bestBid + rand(spreadFailSafe) # If the ask side is non empty place the price 1 tick above the best; otherwise place it random number of ticks above the best bid (since the best ask is zero)
            else # If the bid side is empty (implies the ask side cannot be empty since an error would have thrown)
                price = bestAsk + 1 # Place price 1 tick above best
            end
        else # Aggressive sell LO
            if bestAsk != 0 # If the ask side LOB is non-empty
                price = bestAsk - 1 # Place price 1 tick below best ask even if spread = 1
            else # If the ask side is empty (implies the ask side cannot be empty)
                Random.seed!(seed)
                price = bestBid + rand(spreadFailSafe) # Place price random number of ticks above best bid
            end
        end
    end
    return price
end
#---------------------------------------------------------------------------------------------------

#----- Model 2 -----#
function InjectSimulation(arrivals; seed = 1)
    StartJVM()
    client = Login(1, 1)
    try # This ensures that the client gets logged out whether an error occurs or not
        SubmitOrder(client, Order(arrivals.OrderId[1], arrivals.Side[1], "Limit", arrivals.Volume[1], 1000))
        arrivals.arrivalTime = arrivals.DateTime .+ Time(now())
        previousBestBid = previousBestAsk = 0
        Juno.progress() do id # Progress bar
            for i in 2:nrow(arrivals)
                bestBid = ReceiveMarketData(client, :Bid, :Price); bestAsk = ReceiveMarketData(client, :Ask, :Price)
                if arrivals.Type[i] == :LO # Limit order
                    limitOrder = arrivals[i, :]
                    if bestBid == 0 && bestAsk == 0 # If both sides are empty => implement fail safe
                        println("Both sides of the LOB emptied")
                        price = limitOrder.Side == "Buy" ? SetLimitPrice(limitOrder, bestBid, previousBestAsk, seed) : SetLimitPrice(limitOrder, previousBestBid, bestAsk, seed)
                    else
                        price = SetLimitPrice(limitOrder, bestBid, bestAsk, seed)
                    end
                    limitOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - limitOrder.arrivalTime)) : sleep(limitOrder.arrivalTime - Time(now()))
                    SubmitOrder(client, Order(limitOrder.OrderId, limitOrder.Side, "Limit", limitOrder.Volume, price))
                elseif arrivals.Type[i] == :MO # Market order
                    marketOrder = arrivals[i, :]
                    if (marketOrder.Side == "Buy" && bestAsk != 0) || (marketOrder.Side == "Sell" && bestBid != 0) # Don't submit a trade if the contra side is empty
                        marketOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - marketOrder.arrivalTime)) : sleep(marketOrder.arrivalTime - Time(now()))
                        SubmitOrder(client, Order(marketOrder.OrderId, marketOrder.Side, "Market", marketOrder.Volume))
                    end
                else # Order cancel
                    cancelOrder = arrivals[i, :]
                    if (cancelOrder.Side == "Buy" && bestBid != 0) || (cancelOrder.Side == "Sell" && bestAsk != 0) # Only send a cancel order through if the LOB is non-empty
                        (LOBSnapshot, best) = arrivals.Side[i] == "Buy" ? (ReceiveLOBSnapshot(client, "Buy"), ReceiveMarketData(client, :Bid, :Price)) : (ReceiveLOBSnapshot(client, "Sell"), ReceiveMarketData(client, :Ask, :Price))
                        OrderIds = cancelOrder.Passive ? [k for (k,v) in LOBSnapshot if v.Price != best] : [k for (k,v) in LOBSnapshot if v.Price == best]
                        if !isempty(OrderIds)
                            Random.seed!(seed)
                            orderId = rand(OrderIds) # Passive => sample from orders not in L1; aggressive => sample from orders in L1
                            price = LOBSnapshot[orderId].Price # Get the price of the corresponding orderId
                            cancelOrder.arrivalTime <= Time(now()) ? println(string("Timeout: ", Time(now()) - cancelOrder.arrivalTime)) : sleep(cancelOrder.arrivalTime - Time(now()))
                            CancelOrder(client, orderId, arrivals.Side[i], price)
                        end
                    end
                end
                if bestBid != 0 # Update previous best bid only if it is non-empty
                    previousBestBid = bestBid
                end
                if bestAsk != 0 # Update previous best ask only if it is non-empty
                    previousBestAsk = bestAsk
                end
                seed += 1 # Change seed for next iteration
                @info "Trading" progress=(arrivals.DateTime[i] / arrivals.DateTime[end]) _id=id # Update progress
            end
        end
    finally
        Logout(client)
    end
end
function SetLimitPrice(limitOrder, bestBid, bestAsk, seed)
    spreadFailSafe = collect(1:10) # Vector of ticks from which to sample randomly
    if limitOrder.Side == "Buy" # Buy LO
        if limitOrder.Passive # Passive buy LO
            if bestAsk != 0 # If the ask side LOB is non-empty
                Random.seed!(seed)
                price = bestBid != 0 ? bestBid - 1 : bestAsk - rand(spreadFailSafe) # If the bid side is non empty place the price 1 tick below the best; otherwise place it random number of ticks below the best ask (since the best bid is zero)
            else # If the ask side is empty (implies the bid side cannot be empty since an error would have thrown)
                price = bestBid - 1 # Place price 1 tick below best
            end
        else # Aggressive buy LO
            if bestBid != 0 # If the bid side LOB is non-empty
                spread = abs(bestAsk - bestBid)
                price = spread > 1 ? bestBid + 1 : bestBid # If the spead is greater than 1 tick place price above best; otherwise place price at best (note that this still applies if bestAsk = 0)
            else # If the bid side is empty (implies the ask side cannot be empty)
                Random.seed!(seed)
                price = bestAsk - rand(spreadFailSafe) # Place price random number of ticks below best ask
            end
        end
    else # Sell LO
        if limitOrder.Passive # Passive sell LO
            if bestBid != 0 # If the bid side LOB is non-empty
                Random.seed!(seed)
                price = bestAsk != 0 ? bestAsk + 1 : bestBid + rand(spreadFailSafe) # If the ask side is non empty place the price 1 tick above the best; otherwise place it random number of ticks above the best bid (since the best ask is zero)
            else # If the bid side is empty (implies the ask side cannot be empty since an error would have thrown)
                price = bestAsk + 1 # Place price 1 tick above best
            end
        else # Aggressive sell LO
            if bestAsk != 0 # If the ask side LOB is non-empty
                spread = abs(bestAsk - bestBid)
                price = spread > 1 ? bestAsk - 1 : bestAsk # If the spead is greater than 1 tick place price above best; otherwise place price at best (note that this still applies if bestBid = 0)
            else # If the ask side is empty (implies the ask side cannot be empty)
                Random.seed!(seed)
                price = bestBid + rand(spreadFailSafe) # Place price random number of ticks abive best bid
            end
        end
    end
    return price
end
#---------------------------------------------------------------------------------------------------
#=
simulation = ThinningSimulation(λ₀, α, β, 28800; seed = 5)
reduceLiquid = 0; increaseLiquid = 0
for event in [3; 4; 5; 6]
    increaseLiquid += length(simulation[event])
end
for event in [1; 2; 7; 8; 9; 10]
    reduceLiquid += length(simulation[event])
end
println(increaseLiquid)
println(reduceLiquid)
=#
