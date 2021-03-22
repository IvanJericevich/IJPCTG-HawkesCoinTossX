## Author: Patrick Chang
# Function to simulate a multivariate self exciting point process
# Using the method outlined in Toke and Pomponio 2012.

using LinearAlgebra, Random

#---------------------------------------------------------------------------
### SIMUATION
# The simulation fuction is based on Dharmesh Sing's implementation from Java
#---------------------------------------------------------------------------
## Supporting Functions for the Simulation

# beta = DxD matrix of double floats
# alpha = DxD matrix of double floats

# Returns the Spectral Radius of Γ = A / B
# to check if stability conditions have been met
function SpectralRadius(alpha, beta)
    dimension = size(alpha)[1]
    Γ = zeros(dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if beta[i,j] != 0
                Γ[i,j] = alpha[i,j] / beta[i,j]
            end
        end
    end
    eigenval = eigen(Γ).values
    eigenval = abs.(eigenval)
    return maximum(eigenval)
end

# D = random uniform sample
# I_star = sum(m_lambda) + additional prob of no sample simulated
# m_lambda = vector of intensities for each process at time t

# Returns the index n_0 for the m^{th} process for which
# a sample must be drawn
function attribute(D, I_star, m_lambda)
    index = 1
    cumulative = m_lambda[1]

    while D > (cumulative/I_star)
        index = index + 1
        cumulative += m_lambda[index]
    end
    return index
end

#---------------------------------------------------------------------------
## Simulation Function

# lambda0 = vector of baseline intensities
# alpha = DxD matrix of excitation (double floats)
# beta = DxD matrix of rates of relaxation (double floats)

# Returns vectors of sampled times from the multivariate
# D-type Hawkes process
function simulateHawkes(lambda0, alpha, beta, T; kwargs...)

    kwargs = Dict(kwargs)

    if haskey(kwargs, :seed)
        seed = kwargs[:seed]
    else
        seed = 1
    end

    # Initialize
    dimension = length(lambda0)
    history = Vector{Vector{Float64}}()
    for i in 1:dimension
        history = push!(history, [])
    end

    dlambda = zeros(dimension, dimension)
    m_lambda0 = lambda0
    m_lambda = zeros(dimension, 1)

    SR = SpectralRadius(alpha, beta)
    if SR >= 1
        return println("WARNING: Unstable, Spectral Radius of Γ = A/B must be less than 1")
    end

    lambda_star = 0.0
    t = 0.0

    for i in 1:dimension
        lambda_star += m_lambda0[i]
        m_lambda[i] = m_lambda0[i]
    end

    # First Event
    Random.seed!(seed)
    U = rand()
    s = - log(U)/lambda_star

    if s <= T
        D = rand()
        n0 = attribute(D, lambda_star, m_lambda)
        history[n0] = append!(history[n0], s)

        for i in 1:dimension
            dlambda[i,n0] = alpha[i,n0]
            m_lambda[i] = m_lambda0[i] + alpha[i,n0]
        end
    else
        return history
    end

    t = s

    # General Routine
    lambda_star = 0.0
    for i in 1:dimension
        lambda_star = lambda_star + m_lambda[i]
    end

    while true
        seed += 1
        Random.seed!(seed)

        U = rand()
        s = s - (log(U) / lambda_star)

        seed += 1
        Random.seed!(seed)
        if s <= T
            D = rand()
            I_M = 0.0
            for i in 1:dimension
                dl = 0.0
                for j in 1:dimension
                    dl += dlambda[i,j] * exp(-beta[i,j] * (s - t))
                end
                m_lambda[i] = m_lambda0[i] + dl
                I_M = I_M + m_lambda[i]
            end

            if D <= (I_M / lambda_star)
                n0 = attribute(D, lambda_star, m_lambda)
                history[n0] = append!(history[n0], s)
                lambda_star = 0.0
                for i in 1:dimension
                    dl = 0.0
                    for j in 1:dimension
                        dlambda[i,j] = dlambda[i,j] * exp(-beta[i,j] * (s - t))
                        if n0 == j
                            dlambda[i,n0] += alpha[i,n0]
                        end
                        dl += dlambda[i,j]
                    end
                    lambda_star += m_lambda0[i] + dl
                end
                t = s
            else
                lambda_star = I_M
            end
        else
            return history
        end
    end
end

#---------------------------------------------------------------------------
## Supporting Functions to extract useful information from the Simulation

# Extract the Intensity fuction given the simulation paths
function Intensity(index, time, history, lambda0, alpha, beta)
    l = repeat([lambda0[index]], length(time))
    dimension = length(lambda0)

    for t in 1:length(time)
        for h in 1:length(history[index])
            if time[t] > history[index][h]
                for d in 1:dimension
                    c_alpha = alpha[d,index]
                    c_beta = beta[d,index]

                    l[t] += c_alpha * exp(-c_beta * (time[t] - history[index][h]))
                end
            end
        end
    end
    return l
end

#---------------------------------------------------------------------------
### CALIBRATION
#---------------------------------------------------------------------------
## Supporting functions to calculate R^{nm}(l) used as a Recursion
# outlined in Toke and Pomponio 2012.

# Recursion function to compute R_j^{mn}(l) in the loglikelihood
# for a multivariate Hawkes process
function recursion(history, beta, m, n)
    history_m = history[m]
    history_m = append!([0.0], history_m)
    N = length(history_m)
    R = zeros(Real, N, 1)
    history_n = history[n]
    beta = beta[m,n]
    ix = Int(1)
    for i in 2:N
        if n == m
            R[i] = exp(-beta * (history_m[i] - history_m[i-1])) * (1 + R[i-1])
        else
            R[i] = exp(-beta * (history_m[i] - history_m[i-1])) * R[i-1]
            for j in ix:length(history_n)
                if history_n[j] >= history_m[i-1]
                    if history_n[j] < history_m[i]
                        R[i] += exp(-beta*(history_m[i] - history_n[j]))
                    else
                        ix = j
                        break
                    end
                end
            end
        end
    end
    return R[2:end]
end

# Function to compute the integrated intensity from [0,T]
# ∫_0^T λ^m(t) dt in the loglikelihood
# for a multivariate Hawkes process
function Λ_m(history, T, lambda0, alpha, beta, m)
    Λ = lambda0[m] * T
    dimension = length(lambda0)

    Γ = zeros(Real, dimension, dimension)
    for i in 1:dimension
        for j in 1:dimension
            if beta[i,j] != 0
                Γ[i,j] = Real(alpha[i,j] / beta[i,j])
            end
        end
    end

    for n in 1:dimension
        for i in 1:length(history[n])
            if history[n][i] <= T
                Λ += Γ[m,n] * (1 - exp(-beta[m,n] * (T - history[n][i])))
            end
        end
    end
    return Λ
end

#---------------------------------------------------------------------------
## Functions to compute the log-likelihood to optimize using MLE
# outlined in Toke and Pomponio 2012.

# Computes the partial loglikelihoods and sums them up to obtain the
# full loglikelihood
function loglikeHawkes(history, lambda0, alpha, beta, T)
    dimension = length(lambda0)
    ll = zeros(Real, dimension, 1)

    for m in 1:dimension
        ll[m] = T - Λ_m(history, T, lambda0, alpha, beta, m)
        R_mn = zeros(Real, length(history[m]), dimension)
        for n in 1:dimension
            R_mn[:,n] = recursion(history, beta, m, n)
        end

        ind = findall(x-> x.< 0, R_mn)
        R_mn[ind] .= 0

        for l in 1:length(history[m])
            d = lambda0[m] + sum(alpha[m,:] .* R_mn[l,:])
            if d > 0
                ll[m] += log(d)
            else
                ll[m] += -100
            end
        end
    end
    return sum(ll)
end

#---------------------------------------------------------------------------
### Testing the calibration
#---------------------------------------------------------------------------

# lambda0 = [par[1] par[1]]
# alpha = [0 par[2]; par[2] 0]
# beta = [0 par[3]; par[3] 0]

function GeneralisedResiduals(history, lambda0, alpha, beta)
    # Initialize
    dimension = length(lambda0)
    GE = Vector{Vector{Float64}}()
    for i in 1:dimension
        GE = push!(GE, [])
    end
    # Loop through each dimension
    for m in 1:dimension
        # Initialise the integrated intensity
        Λ = zeros(length(history[m]), 1)
        # Loop through the observations in each process
        for l in 1:length(history[m])
            Λ[l] = Λ_m(history, history[m][l], lambda0, alpha, beta, m)
        end
        # Compute the error and push it into Generalised Errors
        for l in 2:length(history[m])
            # Append results
            GE[m] = append!(GE[m], Λ[l] - Λ[l-1])
        end
    end
    return GE
end
# GE = GeneralisedResiduals(t, lambda0, alpha, beta)
