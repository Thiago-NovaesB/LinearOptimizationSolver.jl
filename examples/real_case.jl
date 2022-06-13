using JuMP, GLPK, LinearAlgebra, Random, HiGHS
using MKL
using LinearOptimizationSolver
using BenchmarkTools, BenchmarkPlots, StatsPlots

optimizer = () -> GLPK.Optimizer()

function build_problem(p::Integer, k::Integer)
    Random.seed!(123)
    e_p = rand(1:5,p)
    cc_p = rand(1:9,p)
    oc_p = rand(1:2,p)
    ic_k = rand(1:30,k)
    dd_k = rand(1:5,k)
    du_k = rand(1:1,k)
    r_k = [dd_k[i]*du_k[i] for i in 1:k]

    lenvars = [p, p*k, k];
    n = sum(lenvars)

    I_pp = Matrix(I,p,p)
    I_kk = Matrix(I,k,k)

    Z_pk = zeros(p,k)
    Z_kp = zeros(k,p)

    D = zeros(p,p*k)
    for i in 1:p
        D[i,1+(i-1)*k:i*k] .= 1
    end

    E = zeros(k,p*k)
    for i in 1:p
        E[:,1+(i-1)*k:i*k] = I_kk
    end

    A = [-I_pp D Z_pk; Z_kp E I_kk; Z_kp -E -I_kk]

    b = vcat(e_p, r_k, -r_k)

    c_y = zeros(p*k)

    for i in 1:p, j in 1:k
        c_y[(i-1)*k+j] = du_k[j]*oc_p[i] 
    end

    c = -[cc_p' c_y' ic_k']

    m = p+2*k

    A = [A Matrix(I,m,m)]
    c = vcat(c',zeros(m))
    return A, b, c, n, m
end

p = 30
k = 10
A, b, c, n, m = build_problem(p, k)

input = create(A, b, c, solver = 1, verbose=false) 
bench1 = @benchmark output = solve(input)
input = create(A, b, c, solver = 0, verbose=false) 
bench2 = @benchmark output = solve(input)

function GLPK!(A, b, c)
    model_pl = Model(GLPK.Optimizer);
    m = size(A,1);
    n = size(A,2) - length(b);
    @variable(model_pl, X[1:n + m] >= 0);
    @constraint(model_pl, A*X .== b);
    @objective(model_pl, Max, c'X);
    optimize!(model_pl);
end

function HiGHS!(A, b, c)
    model_pl = Model(HiGHS.Optimizer);
    set_silent(model_pl)
    m = size(A,1);
    n = size(A,2) - length(b);
    @variable(model_pl, X[1:n + m] >= 0);
    @constraint(model_pl, A*X .== b);
    @objective(model_pl, Max, c'X);
    optimize!(model_pl);
end

bench3 = @benchmark GLPK!($A, $b, $c)
bench4 = @benchmark HiGHS!($A, $b, $c)

plotd = plot(bench1,yaxis=:log10,st=:violin)
plot!(bench2,yaxis=:log10,st=:violin,xticks=(1:2,["IP" "Simplex"]))
plot!(bench3,yaxis=:log10,st=:violin,xticks=(1:3,["IP" "Simplex" "GLPK"]))
plot!(bench4,yaxis=:log10,st=:violin,xticks=(1:4,["IP" "Simplex" "GLPK" "HiGHS"]))

savefig(plotd,"examples\\p=$(p),k=$(k).png")