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
#---------------------------------------------------------------------------------------------------

#----- Simulation functions -----#
function InjectSimulation(arrivals; seed = 1)
    Random.seed!(seed)
    StartJVM()
    client = Login(1, 1)
    bids = Dict{String, Int64}(); asks = Dict{String, Int64}()
    try # This ensures that the client gets logged out whether an error occurs or not
        SubmitOrder(client, Order(arrivals.OrderRef[1], arrivals.Side[1], "Limit", Int(arrivals.Quantity[1]), Int(arrivals.Price[1])))
        WaitForMarketData(client)
        arrivals.arrivalTime = arrivals.DateTime .+ Time(now())
        Juno.progress() do id # Progress bar
            for i in 2:nrow(arrivals)
                # Validation
                if isempty(bids) || isempty(asks)
                    println("Broken LOB")
                end
                if arrivals.Type[i] == :LO
                    arrivals.arrivalTime[i] <= Time(now()) ? println("Timeout") : sleep(arrivals.arrivalTime[i] - Time(now()))
                    SubmitOrder(client, Order(orderId, side, type, volume, price))

                elseif arrivals.Type[i] == :MO
                    arrivals.arrivalTime[i] <= Time(now()) ? println("Timeout") : sleep(arrivals.arrivalTime[i] - Time(now()))
                    SubmitOrder(client, Order(arrivals.OrderId[i], arrivals.Side[i], "Market", arrivals.Volume[i]))
                else
                    if arrivals.Side == "Buy"
                        if arrivals.Passive[i]

                        else
                        end
                    else
                    end



                    orderId =
                    price =

                    arrivals.arrivalTime[i] <= Time(now()) ? println("Timeout") : sleep(arrivals.arrivalTime[i] - Time(now()))
                    CancelOrder(client, orderId, arrivals.Side[i], price)
                end
                WaitForMarketData(client)
                @info "Trading" progress=(arrivals.DateTime[i] / arrivals.DateTime[end]) _id=id # Update progress
            end
        end
    finally
        Logout(client)
    end
end
function RPareto(xn, α, n = 1)
    return xn ./ ((rand(n)).^(1/α))
end
#---------------------------------------------------------------------------------------------------

#----- Implementation -----#
Random.seed!(5)
λ₀ = [0.01; 0.01; 0.02; 0.02; 0.02; 0.02; 0.015; 0.015; 0.015; 0.015]
β  = fill(0.2, 10, 10)
α  = [repeat([0.01], 10)'; repeat([0.01], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.02], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)'; repeat([0.015], 10)']
arrivals = simulateHawkes(λ₀, α, β, 300; seed = 5) |> x -> DataFrame(DateTime = reduce(vcat, x),Type = reduce(vcat, fill.([:MO, :MO, :LO, :LO, :LO, :LO, :OC, :OC, :OC, :OC], length.(x))), Side = reduce(vcat, fill.(["Buy", ":Sell", "Buy", ":Sell", "Buy", ":Sell", "Buy", ":Sell", "Buy", ":Sell"], length.(x))),
OrderId = vcat(string.(collect(1:sum(length.(x[1:6])))), fill("0", sum(length.(x[7:10])))), Volume = round.(Int, vcat(RPareto(20, 1.5, sum(length.(x[1:2]))), RPareto(20, 1, sum(length.(x[3:6]))), fill(0, sum(length.(x[7:10]))))),
Passive = reduce(vcat, fill.([false, false, false, false, true, true, false, false, true, true], length.(x))))
sort!(arrivals, :DateTime)
arrivals.DateTime .-= arrivals.DateTime[1]
delete!(arrivals, [1, 2])
InjectSimulation(arrivals)
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
=#
