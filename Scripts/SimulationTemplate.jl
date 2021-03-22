function InjectSimulation(arrivals; seed = 1)
    Random.seed!(seed)
    StartJVM()
    client = Login(1, 1)
    try # This ensures that the client gets logged out whether an error occurs or not
        SubmitOrder(client, Order(arrivals.OrderRef[1], arrivals.Side[1], "Limit", Int(arrivals.Quantity[1]), Int(arrivals.Price[1])))
        WaitForMarketData(client)
        arrivals.arrivalTime = arrivals.Date .+ Time(now())
        Juno.progress() do id # Progress bar
            for i in 2:nrow(arrivals)
                arrivals.arrivalTime[i] <= Time(now()) ? println("Timeout") : sleep(arrivals.arrivalTime[i] - Time(now()))    
                SubmitOrder(client, Order(orderId, side, type, volume, price)) # Limit order
                SubmitOrder(client, Order(orderId, side, type, volume)) # Market order
                CancelOrder(client, orderId, side, price) # Cancel order
                WaitForMarketData(client)
                @info "Trading" progress=(arrivals.Date[i] / arrivals.Date[end]) _id=id # Update progress
            end
        end
    finally
        Logout(client)
    end
end
