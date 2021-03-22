## Author: Patrick Chang & Ivan Jericevich
# Script file to simulate events for a 10 variate Hawkes process
# to pass through CoinTossX

# Event types include:
# 1: MO to buy
# 2: MO to sell
# 3: Aggressive LO to buy
# 4: Aggressive LO to sell
# 5: Passive LO to buy
# 6: Passive LO to sell
# 7: Aggressive cancellation to buy
# 8: Aggressive cancellation to sell
# 9: Passive cancellation to buy
# 10: Passive cancellation to sell

# Main thing to check here is that our parameters result in a relatively balanced system
# where the number of events reducing the liquidity is roughly the same as the
# number of events increasing the liquidity.
# i.e. ∑(3+4+5+6) ≈ ∑(1+2+7+8+9+10). The event counts of these should be roughly the same.


#---------------------------------------------------------------------------
## Preamble
using ProgressMeter

include("Hawkes.jl")

cd("/Users/patrickchang1/IJPCDHTG-CoinTossX")

#---------------------------------------------------------------------------
## Parameter setup

# λ₀ = repeat([0.015], 10)
λ₀ = [0.01; 0.01; 0.02; 0.02; 0.02; 0.02; 0.015; 0.015; 0.015; 0.015]
β  = fill(0.2, 10, 10)
α  = [repeat([0.01], 10)';
      repeat([0.01], 10)';
      repeat([0.02], 10)';
      repeat([0.02], 10)';
      repeat([0.02], 10)';
      repeat([0.02], 10)';
      repeat([0.015], 10)';
      repeat([0.015], 10)';
      repeat([0.015], 10)';
      repeat([0.015], 10)']

# # Testing
# test = sum(α ./ β, dims = 2)
#
# sum(test[3:6])
#
# sum(test[1:2]) + sum(test[7:10])
#
# SpectralRadius(α, β)

T = 8*3600
sim = simulateHawkes(λ₀, α, β, T; seed = 5)

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
