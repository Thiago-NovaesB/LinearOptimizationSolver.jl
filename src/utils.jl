function create(A::Matrix{}, b::Vector{}, c::Vector{};
                standard::Bool = true, rho::Float64 = 0.05, alpha::Float64 = 0.95, tol::Float64 = 1E-6, max_iter::Int = 1000,
                verbose::Bool = true, solver::Int = 0, crossover::Bool = false, crossover_tol::Float64 = 1E-3)
        if standard
            m = length(b)
            n = length(c)
        else
            m = length(b)
            extra = zeros(m,m)
            for i in 1:m
                extra[i,i] = 1.0
            end
            A = [A extra]
            c = [c; zeros(m)]
            n = length(c)
        end
        input = Input(A,b,c,standard,n,m,rho,alpha,tol,max_iter,verbose,solver,crossover, crossover_tol)
    return input
end

function update_midterm!(input::Input, midterm::MidTerm)
    A = input.A
    b = input.b
    c = input.c
    base = midterm.base
    B = view(A,:,base)
    x = B \ b
    midterm.z = c[base]'x

    x_opt = zeros(input.n)
    x_opt[base] = x

    midterm.x = x_opt
    nothing
end

function write_output(input::Input, midterm::MidTerm)

    if midterm.termination_status == 2
        output = OutputSimplex(midterm.d, Inf, midterm.termination_status, midterm.base, midterm.nbase)
        last_log(input, output)
    elseif midterm.termination_status == 1
        output = OutputSimplex(midterm.x, midterm.z, midterm.termination_status, midterm.base, midterm.nbase)
        last_log(input, output)
    elseif midterm.termination_status == 3
        output = OutputSimplex(midterm.x, midterm.z, midterm.termination_status, midterm.base, midterm.nbase)
        last_log(input, output)
    elseif midterm.termination_status == 0
        output = OutputSimplex(midterm.x, midterm.z, midterm.termination_status, midterm.base, midterm.nbase)
        last_log(input, output)
    end
    return output
end

function solve(input::Input)
    if input.solver == 0
        output = solve_simplex(input)
    elseif input.solver == 1
        output = solve_ip(input)
        if input.crossover && output.termination_status == 1
            output = crossover(input, output)
        end 
    else
        error("Use solver = 0 para Simplex e solver = 1 para Pontos interiores.")
    end
    return output
end