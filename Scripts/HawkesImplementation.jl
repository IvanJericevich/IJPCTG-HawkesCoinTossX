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
using DataFrames, Dates, Optim#, HypothesisTests
clearconsole()
include("Hawkes.jl")
include("DataCleaning.jl")
include("CoinTossXUtilities.jl")
function RPareto(xn, α, n = 1)
    return xn ./ (rand(n) .^ (1 / α))
end
#---------------------------------------------------------------------------------------------------

#----- Simulation -----#
Random.seed!(5)
λ₀ = [0.01; 0.01; 0.02; 0.02; 0.02; 0.02; 0.015; 0.015; 0.015; 0.015]
α = reduce(hcat, fill(λ₀, 10)) # α  = [repeat([0.01], 10)'; repeat([0.01], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)']
β  = fill(0.2, 10, 10)
arrivals = ThinningSimulation(λ₀, α, β, 1000; seed = 5) |> x -> DataFrame(DateTime = map(t -> Millisecond(round(Int, t * 1000)), reduce(vcat, x)),Type = reduce(vcat, fill.([:MO, :MO, :LO, :LO, :LO, :LO, :OC, :OC, :OC, :OC], length.(x))), Side = reduce(vcat, fill.(["Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell"], length.(x))),
Volume = round.(Int, vcat(RPareto(20, 1.5, sum(length.(x[1:2]))), RPareto(20, 1, sum(length.(x[3:6]))), fill(0, sum(length.(x[7:10]))))), Passive = reduce(vcat, fill.([false, false, false, false, true, true, false, false, true, true], length.(x))))
sort!(arrivals, :DateTime)
delete!(arrivals, 1)
arrivals.DateTime .-= arrivals.DateTime[1]
indeces = findall(x -> x < Millisecond(10), diff(arrivals.DateTime))
delete!(arrivals, indeces .+ 1)
arrivals.OrderId = string.(collect(1:nrow(arrivals)))
InjectSimulation(arrivals, seed = 5)
#---------------------------------------------------------------------------------------------------

#----- Hawkes Recalibration -----#
events = [(:WalkingMO, :Buy, true), (:WalkingMO, :Sell, true), (:LO, :Buy, true), (:LO, :Sell, true), (:LO, :Buy, false), (:LO, :Sell, false), (:OC, :Buy, true), (:OC, :Sell, true), (:OC, :Buy, false), (:OC, :Sell, false)]
data = PrepareData("OrdersSubmitted_2", "Trades_2") |> x -> ClassifyHawkesEvents(x) |> y -> groupby(y, [:Type, :Side, :IsAggressive]) |> z -> map(event -> Dates.value.(collect(z[event].Time)), events)
initialSolution = vec(vcat(λ₀, reshape(α, :, 1), reshape(β, :, 1)))
logLikelihood = TwiceDifferentiable(θ -> Calibrate(θ, data, 300, 10), initialSolution, autodiff = :forward)
@time calibratedParameters = optimize(logLikelihood, initialSolution, LBFGS(), Optim.Options(show_trace = true))
#=
open("Parameters.txt", "w") do file
    for p in exp.(Optim.minimizer(calibratedParameters))
        println(file, p)
    end
end
=#
#---------------------------------------------------------------------------------------------------

#----- Hawkes calibration validation -----#
#=
for i in 1:length(events)
    # QQ-plots
    integratedIntensities = Λ(i, θ[1], θ[2], θ[3], simulatedArrivals)
    qqPlot = qqplot(Exponential(1), integratedIntensities, xlabel = "Exponential theoretical quantiles", ylabel = "Sample quantiles", title = titles[i], markersize = 3, linecolor = :black, markercolor = col[i], markerstrokecolor = col[i], legend = false)
    savefig(qqPlot, string("LBFGS/QQPlot - ", events[i][1], ".pdf"))
    # Independence plots
    Uᵢ = cdf.(Exponential(1), integratedIntensities)[1:(end - 1)] # diff(simulatedArrivals[i])
    Uᵢ₊₁ = cdf.(Exponential(1), integratedIntensities)[2:end]
    independencePlot = plot(Uᵢ, Uᵢ₊₁, seriestype = :scatter, markersize = 3, markercolor = col[i], markerstrokecolor = col[i], title = titles[i], legend = false) # , xlabel = L"U_k = F_{Exp(1)}(t_k - t_{k - 1})", ylabel = L"U_{k + 1} = F_{Exp(1)}(t_{k + 1} - t_k)"
    savefig(independencePlot, string("LBFGS/Independence Plot - ", events[i][1], ".pdf"))
    # Statistical tests
    LBTest = LjungBoxTest(integratedIntensities, 20, 3) # Ljung-Box - H_0 = independent
    println(file, string(events[i][1], "-LB,", round(LBTest.Q, digits = 5), ",", round(pvalue(LBTest), digits = 5)))
    KSTest = ExactOneSampleKSTest(integratedIntensities, Exponential(1)) # KS - H_0 = exponential
    println(file, string(events[i][1], "-KS,", round(KSTest.δ, digits = 4), ",", round(pvalue(KSTest), digits = 4)))
end
=#
#---------------------------------------------------------------------------------------------------

#----- Model 1 -----#
function InjectSimulation(arrivals; seed = 1)
    Random.seed!(seed)
    StartJVM()
    client = Login(1, 1)
    try # This ensures that the client gets logged out whether an error occurs or not
        SubmitOrder(client, Order("1", "Buy", "Limit", round(Int, RPareto(20, 1)[1]), 45))
        SubmitOrder(client, Order("2", "Buy", "Limit", round(Int, RPareto(20, 1)[1]), 45))
        SubmitOrder(client, Order("3", "Buy", "Limit", round(Int, RPareto(20, 1)[1]), 44))
        SubmitOrder(client, Order("4", "Sell", "Limit", round(Int, RPareto(20, 1)[1]), 50))
        SubmitOrder(client, Order("5", "Sell", "Limit", round(Int, RPareto(20, 1)[1]), 50))
        SubmitOrder(client, Order("6", "Sell", "Limit", round(Int, RPareto(20, 1)[1]), 51))
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
        SubmitOrder(client, Order(arrivals.OrderId[1], arrivals.Side[1], "Limit", arrivals.Volume[1], 51))
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
                    SubmitOrder(client, Order(limitOrder.OrderId, limitOrder.Side, "Limit", limitOrder.Volume, price))
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

#=
reduceLiquid = 0; increaseLiquid = 0
for event in [3; 4; 5; 6]
    reduceLiquid += length(simulation[event])
end
for event in [1; 2; 7; 8; 9; 10]
    increaseLiquid += length(simulation[event])
end
println(reduceLiquid)
println(increaseLiquid)
=#
