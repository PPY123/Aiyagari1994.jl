include("am.jl")

using BenchmarkTools

# Create an instance of Household
am = Household(r = 0.03, β = 0.96, a_size = 500)
a_size = am.a_size
V, g_a, g_c, g_h = initialize(am)

println("===============")
println("TIME:")
println("===============")
println("VFI - Discrete with no refinements. Grid size = $a_size")
@btime compute_fixed_point(v -> BruteForce_Std(v, am),
                        V,
                        max_iter=1000,
                        err_tol=1e-4,
                        verbose=false)
println("===============")
println("VFI - Discrete using Monotonicity. Grid size = $a_size")
@btime compute_fixed_point(v -> BruteForce_Mtcy(v, am),
                        V,
                        max_iter=1000,
                        err_tol=1e-4,
                        verbose=false)
println("===============")
println("VFI - Continous using linear interpolation. Grid size = $a_size")
@btime compute_fixed_point(v -> BellmanOperator(v, am),
                        V,
                        max_iter=1000,
                        err_tol=1e-4,
                        verbose=false)
println("===============")
println("PFI - Continous using linear interpolation. Grid size = $a_size")
@btime compute_fixed_point(c -> ColemanOperator(c, am),
                        g_c,
                        max_iter=1000,
                        err_tol=1e-4,
                        verbose=false)
println("===============")
println("Endogeous Grid Method. Grid size = $a_size")
@btime compute_fixed_point(c -> EGMOperator_Labor(c, am),
                        g_c,
                        max_iter=1000,
                        err_tol=1e-4,
                        verbose=false)
