using JuMP, GLPK, LinearAlgebra, Random, Distributions
using LinearOptimizationSolver
using BenchmarkTools, BenchmarkPlots, StatsPlots
using Plots

function build_random(p::Int64)
    Random.seed!(123)
    A_prime = rand(Uniform(0, 1),p,p)
    A = hcat(A_prime,I)
    c = zeros(2*p)
    c[1:p] = rand(Uniform(0, 1),p)
    b = rand(Uniform(1, 2),p)
    return A, b, c, 2*p, p
end
p = 100
A, b, c, n, m = build_random(p)

input = create(A, b, c, solver = 1, verbose=false) 
bench1 = @benchmark output = solve(input)
input = create(A, b, c, solver = 1, verbose=false,crossover = true) 
bench2 = @benchmark output = solve(input)
input = create(A, b, c, solver = 0, verbose=false) 
bench3 = @benchmark output = solve(input)

function GLPK!(A, b, c)
    model_pl = Model(GLPK.Optimizer);
    m = size(A,1);
    n = size(A,2) - length(b);
    @variable(model_pl, X[1:n + m] >= 0);
    @constraint(model_pl, A*X .== b);
    @objective(model_pl, Max, c'X);
    optimize!(model_pl);
end

bench4 = @benchmark GLPK!($A, $b, $c)

plotd = plot(bench1,yaxis=:log10,st=:violin)
plot!(bench2,yaxis=:log10,st=:violin,xticks=(1:2,["IP" "IP Crossover"]))
plot!(bench3,yaxis=:log10,st=:violin,xticks=(1:3,["IP" "IP Crossover" "Simplex"]))
plot!(bench4,yaxis=:log10,st=:violin,xticks=(1:4,["IP" "IP Crossover" "Simplex" "GLPK"]))

savefig(plotd,"examples\\n=$(p).png")