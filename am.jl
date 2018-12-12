using Parameters, QuantEcon, LinearAlgebra, Interpolations, Random
using Optim, Roots

Household = @with_kw (r = 0.01,
                      R = 1 + r,
                      α = 0.33,
                      δ = 0.05,
                      w = (1-α)*(α/(r+δ))^(α/(1-α)),
                      σ = 1.0,
                      κ = 4.00,
                      ν = 0.25,
                      β = 0.96,
                      MC = MarkovChain([0.9 0.1; 0.1 0.9], [0.95; 1.05]),
                      Π = MC.p,
                      z_vals = MC.state_values,
                      z_size = length(z_vals),
                      b = 0.0, # Borrowing limit/Minimum level of assets
                      r_ss = (1/β-1), # SS interest rate
                      a_max = ((r_ss+δ)/α)^(1/(α-1))*3, # 3 times SS capital
                      a_size = 200,
                      a_vals = collect(range(b, a_max, length = a_size)),
                      u = σ == 1 ? (x, h=0.0) -> log(x) -  κ*(h^1+ν)/(1+ν) : (x, h=0.0) -> (x^(1 - σ) - 1) / (1 - σ) - κ*(h^1+ν)/(1+ν),
                      duc = c -> c^(-σ),
                      duh = h -> κ*h^(ν),
                      duc_inv = x -> x^(-1/σ),
                      duh_inv = x -> (x/κ)^(1/ν))

function BruteForce_Mtcy(V::AbstractArray,
                            am::NamedTuple;
                            return_policy::Bool=false)

    @unpack R, r, w, b, σ, β, Π, z_vals, a_size, a_vals, z_size, duc, duh, duc_inv, duh_inv, u = am
    U = fill(-Inf, a_size, z_size, a_size)
    V_new = similar(V)
    g_a = zeros(Int, a_size, z_size)

    for i_a in 1:a_size
        for i_z in 1:z_size
            if i_a == 1
                a_new_min = 1
            else
                a_new_min = g_a[i_a-1, i_z]
            end
            a_new_max = min(a_new_min + 15, a_size)
            for i_a_new in a_new_min:a_new_max
                c = w * z_vals[i_z] + R * a_vals[i_a] - a_vals[i_a_new]
                if c > 0.0
                    U[i_a, i_z, i_a_new] = u(c) + β*dot(Π[i_z,:],V[i_a_new,:])
                end
            end
            V_new[i_a, i_z], g_a[i_a, i_z] = findmax(U[i_a, i_z, :])
        end
    end
    if return_policy == true
        return V_new, g_a
    else
        return V_new
    end
end

function BruteForce_Std(V::AbstractArray,
                            am::NamedTuple;
                            return_policy::Bool=false)

    @unpack R, r, w, b, σ, β, Π, z_vals, a_size, a_vals, z_size, duc, duh, duc_inv, duh_inv, u = am
    U = fill(-Inf, a_size, z_size, a_size)
    V_new = similar(V)
    g_a = similar(V)

    for i_a in 1:a_size
        for i_z in 1:z_size
            for i_a_new in 1:a_size
                c = w * z_vals[i_z] + R * a_vals[i_a] - a_vals[i_a_new]
                if c > 0
                    U[i_a, i_z, i_a_new] = u(c) + β*dot(Π[i_z,:],V[i_a_new,:])
                end
            end
            V_new[i_a, i_z], g_a[i_a, i_z] = findmax(U[i_a, i_z, :])
        end
    end
    if return_policy == true
        return V_new, g_a
    else
        return V_new
    end
end

function BellmanOperator(V::AbstractArray,
                            am::NamedTuple;
                            return_policy::Bool=false)

    @unpack R, r, w, b, σ, β, Π, z_vals, a_size, a_vals, z_size, duc, duh, duc_inv, duh_inv, u = am
    vf = interp(a_vals, V)
    z_idx = 1:z_size
    V_new = similar(V)
    g_a = similar(V)
    g_c = similar(V)

    for (i_a, a) in enumerate(a_vals)
        for (i_z, z) in enumerate(z_vals)
            function obj(x)
                EV = dot(Π[i_z, :], vf(R * a + w * z - x, z_idx)) # compute expectation
                return  u(x) +  β * EV
            end
            res = Optim.maximize(obj, 0.0, R * a + w * z + b)
            Optim.converged(res) || error("Didn't converge") # important to check
            g_c[i_a, i_z] = Optim.maximizer(res)
            g_a[i_a, i_z] = R * a + w * z - g_c[i_a, i_z]
            V_new[i_a, i_z] = Optim.maximum(res)
        end
    end
    if return_policy
        V_new, g_a, g_c
    else
        V_new
    end
end

function ColemanOperator(g::AbstractArray,
                            am::NamedTuple;
                            return_policy::Bool=false)
    @unpack R, r, w, b, σ, β, Π, z_vals, a_size, a_vals, z_size, duc, duh, duc_inv, duh_inv, u = am
    gf = interp(a_vals, g)
    z_idx = 1:z_size
    V_new = similar(g)
    g_a = similar(g)
    g_c = similar(g)
    opt_lb = 1e-8
    for (i_a, a) in enumerate(a_vals)
        for (i_z, z) in enumerate(z_vals)
            function obj(x)
                cps = gf.(R * a + z*w - x, z_idx) # c' for each z'
                expectation = dot(duc.(cps), Π[i_z, :])
                return abs(duc(x) - max(R*β * expectation, duc(R * a + z*w + b)))
            end
            opt_ub = R*a + z*w + b  # addresses issue #8 on github
            res = optimize(obj, min(opt_lb, opt_ub - 1e-2), opt_ub,
                           method = Optim.Brent())
            g_c[i_a, i_z] = Optim.minimizer(res)
            g_a[i_a, i_z] = R * a + z*w - g_c[i_a, i_z]
        end
    end
    V_new = u.(g_c)
    if return_policy
        V_new, g_a, g_c
    else
        g_c
    end
end

function EGMOperator(Kg_f::Function,
                            am::NamedTuple;
                            return_policy::Bool=false)

    @unpack R, r, w, b, σ, β, Π, z_vals, a_size, a_vals, z_size, duc, duh, duc_inv, duh_inv, u = am
    g_a = zeros(Float64, a_size, z_size)
    g_c_nb = zeros(Float64, a_size, z_size)
    g_c_new = zeros(Float64, a_size, z_size)

    for (i_z, z) in enumerate(z_vals)
        for (i_a, a_new) in enumerate(a_vals)
            expectation = dot(duc.(g_c[i_a, :]), Π[i_z, :])
            g_c_nb[i_a, i_z] = duc_inv(R*β * expectation)
            g_a[i_a, i_z] = (g_c_nb[i_a, i_z] + a_new - w*z) /R
        end
        # Update policy function and return
        g_c_new[:, i_z] = LinearInterpolation(g_a[:,i_z], g_c_nb[:,i_z], extrapolation_bc=Line())(a_vals)
    end
    for (i_a, a_new) in enumerate(a_vals)
        for (i_z, z) in enumerate(z_vals)
            if a_new < g_a[1, i_z]
                g_c_new[i_a, i_z] = a_new*R + duh_inv(w*z*duc_inv(c))*w*z + b
            end
        end
    end
    return g_c_new
end

function EGMOperator_Labor(g_c::AbstractArray,
                            am::NamedTuple;
                            return_policy::Bool=false)

    @unpack R, r, w, b, σ, β, Π, z_vals, a_size, a_vals, z_size, duc, duh, duc_inv, duh_inv, u = am
    # Initialize arrays
    g_a = zeros(Float64, a_size, z_size)
    g_c_nb = zeros(Float64, a_size, z_size)
    g_c_new = zeros(Float64, a_size, z_size)

    # Iterate over different combinations of (a',z)
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a_new) in enumerate(a_vals)
            # Solve for C(a',z) in the no binding case
            expectation = dot(duc.(g_c[i_a, :]), Π[i_z, :])
            g_c_nb[i_a, i_z] = duc_inv(R*β * expectation)
            # Solve for H(c,z)
            h = duh_inv(w*z*duc_inv(g_c_nb[i_a, i_z]))
            # Solve for A(a',z)
            g_a[i_a, i_z] = (g_c_nb[i_a, i_z] + a_new - h*w*z) /R
        end
        # Update to c1(a',z')
        g_c_new[:, i_z] = LinearInterpolation(g_a[:,i_z], g_c_nb[:,i_z], extrapolation_bc=Line())(a_vals)
    end

    # Iterate over different combinations of (a',z)
    for (i_a, a_new) in enumerate(a_vals)
        for (i_z, z) in enumerate(z_vals)
            # Check if it is a binding borrowing constraint case
            if a_new < g_a[1, i_z]
                # If so, solve for C(a',z) using a root finder and use the result value in c1(a',z')
                cf = c -> abs(a_new*R + duh_inv(w*z*duc_inv(c))*w*z + b - c)
                res = optimize(cf, 0.0, g_c_new[i_a, i_z])
                g_c_new[i_a, i_z] = Optim.minimizer(res)
            end
        end
    end
    # Return c1(a',z')
    return g_c_new
end

function initialize(am::NamedTuple)
    # simplify names, set up arrays
    @unpack R, r, w, b, σ, β, Π, z_vals, a_size, a_vals, z_size, duc, duh, duc_inv, duh_inv, u = am
    V0 = zeros(Float64, a_size, z_size)
    c0 = zeros(Float64, a_size, z_size)
    a0 = [a_vals a_vals]
    h0 = zeros(Float64, a_size, z_size)

    # Populate V and c
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(a_vals)
            c_max = R * a + w*z + b
            c0[i_a, i_z] = c_max
            V0[i_a, i_z] = u(c_max) / (1 - β)
        end
    end
    return V0, a0, c0, h0
end

function SimulationAssets(am::NamedTuple,
                    g_c::AbstractArray,
                    T::Int = 100000;
                    labor::Bool = false,
                    verbose::Bool = false)

    @unpack Π, r, w, α, δ, a_vals, z_vals, R, duc_inv, duh_inv = am  # Simplify names

    cf = interp(a_vals, g_c)

    a_sim = zeros(Float64, T + 1)
    h_sim = zeros(Float64, T)
    z_sim = simulate(MarkovChain(Π), T)
    for t in 1:T
        i_z = z_sim[t]
        z = z_vals[i_z]
        if labor
            h_sim[t] = duh_inv(w*z*duc_inv(cf(a_sim[t], i_z)))
            a_sim[t+1] = R * a_sim[t] + h_sim[t]*w*z - cf(a_sim[t], i_z)
        else
            a_sim[t+1] = R * a_sim[t] + w*z - cf(a_sim[t], i_z)
        end
    end
    # Burn 1000 first periods
    a_sim = a_sim[1000:end]

    # Compute Aggregates from market clearin
    K = mean(a_sim)
    L = mean(z_sim.*h_sim)
    r_new = α*(K/L)^(α-1) - δ
    return a_sim, K, L, r_new
end
