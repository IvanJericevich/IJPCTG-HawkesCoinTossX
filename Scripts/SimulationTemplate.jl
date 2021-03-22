function InjectSimulation1(arrivals; seed = 1)
    Random.seed!(seed)
    StartJVM()
    client = Login(1, 1)
    try # This ensures that the client gets logged out whether an error occurs or not
        SubmitOrder(client, Order("Buy", "Limit", volume, 40)); SubmitOrder(client, Order("Sell", "Limit", volume, 60)) # Initialize LOB
        Juno.progress() do id # Progress bar
            arrivals.DateTime .+= now()
            for i in 1:nrow(arrivals)
                @time WaitForMarketData(client)
                bestBid = ReceiveMarketData(client, :Bid, :Price); bestAsk = ReceiveMarketData(client, :Ask, :Price) # Request market data to update L1LOB each time an order is sent
                arrivals.DateTime[i] > now() ? continue : sleep(arrivals.DateTime[i] - now()) # Skip the order if market data retrieval took too long else wait until the arrival time of the next order
                SubmitOrder(client, Order(orderId, side, type, volume, price)) # Limit order
                SubmitOrder(client, Order(orderId, side, type, volume)) # Market order
                CancelOrder(client, orderId, side, price)
                @info "Trading" progress=(arrivals.DateTime[i] / arrivals.DateTime[end]) _id=id # Update progress
            end
        end
    finally
        Logout(client)
    end
end
