include("am.jl")

using Plots
gr()

function SolveEquilibrium(r0; verbose=false)
    am = Household(r = r0,
                          α = 0.33,
                          δ = 0.05,
                          σ = 3/2,
                          κ = 1.00,
                          ν = 2/3,
                          β = 0.97,
                          MC = MarkovChain([0.5 0.5; 0.5 0.5], [0.95; 1.05]),
                          b = 0.0,
                          a_max = 45,
                          a_size = 250)

    @unpack R, α, δ, r, r_ss, w, b, σ, β, Π, z_vals, a_size, a_vals, z_size, duc, duh, duc_inv, duh_inv, u = am
    V, g_a, g_c, g_h = initialize(am)

    g_c = compute_fixed_point(c -> EGMOperator_Labor(c, am),
                            g_c,
                            max_iter=1000,
                            err_tol=1e-4,
                            verbose=false,
                            print_skip = 100)
    Random.seed!(42)
    a_sim, K, L, r1 = SimulationAssets(am, g_c, labor=true)
    residual = (r1 - r0)/r0
    if verbose
        println("Solving Aiyagari Model for r = $r:")
        println("Endogeous Grid Method. Grid size = $a_size")
        println("Given r0 = $r0, then K = $K and L = $L, which implies r1 = $r1")
        println("Diff = $residual")
    end
    return residual, am, a_sim, g_c
end

r_star = find_zero(SolveEquilibrium, (0.01, 0.05))
_, am, a_sim, g_c = SolveEquilibrium(r_star)
histogram(a_sim, nbins = 100, leg = false, normed = true, xlabel = "assets")
plot(am.a_vals, g_c)
