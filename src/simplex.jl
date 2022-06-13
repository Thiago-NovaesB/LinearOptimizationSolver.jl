function solve_simplex(input::Input)

    midterm = MidTerm()
    if check_phase_1(input)
        midterm.base =  collect((input.n - input.m + 1):input.n)
        midterm.nbase = collect(1:(input.n-input.m))
        val, i = findmin(input.b)

        aux = midterm.base[i]
        push!(midterm.nbase, aux)
        midterm.base[i] = input.n + 1
        c_mem = copy(input.c)
        input.A = [input.A -ones(input.m)] 
        input.c = zeros(input.n + 1)
        input.c[end] = -1
        input.n = input.n + 1

        init_log_simplex1(input)
        while midterm.termination_status == 0 && midterm.iter < input.max_iter
            iterate!(input, midterm)
        end
        update_midterm!(input, midterm)
        if midterm.z < -input.tol
            midterm.termination_status = 3 #infeasible
            output = write_output(input, midterm)
            return output
        end
        midterm.iter = 0
        input.n = input.n - 1
        cache = findfirst(x->x==input.n + 1,midterm.nbase)
        input.c = c_mem
        midterm.termination_status = 0

        if cache !== nothing
            deleteat!(midterm.nbase, cache)
        else
            l = findfirst(x->x==input.n + 1,midterm.base)
            B = view(input.A,:,midterm.base)
            for (k,w) in enumerate(midterm.nbase)
                sol = B \ input.A[:,w]
                if abs.(sol[l]) > input.tol
                    push!(midterm.base, w)
                    deleteat!(midterm.nbase, k)
                    deleteat!(midterm.base, l)
                    break
                end
                if k == length(midterm.nbase)
                    input.A = input.A[1:end .!= l, :]
                    input.b = input.b[1:end .!= l]
                    deleteat!(midterm.base, l)
                    input.m -= 1
                    break
                end
            end
        end
        input.A = input.A[:,1:input.n]
    else
        midterm.base =  collect((input.n - input.m + 1):input.n)
        midterm.nbase = collect(1:(input.n-input.m))
    end
    
    
    init_log_simplex2(input)
    while midterm.termination_status == 0 && midterm.iter < input.max_iter
        iterate!(input, midterm)
    end
    update_midterm!(input, midterm)
    output = write_output(input, midterm)
    return output
end

function check_phase_1(input::Input)
    b_min = minimum(input.b)
    return b_min < 0 
end

function iterate!(input::Input, midterm::MidTerm)

    midterm.iter += 1
    A = input.A
    b = input.b
    c = input.c
    base = midterm.base
    nbase = midterm.nbase
    tol = input.tol
    B = view(A,:,base)
    N = view(A,:,nbase)
    xB = B \ b
    y = B' \ c[base]
    midterm.red_cost = c[nbase] - N'*y
    val = maximum(midterm.red_cost)
    
    if val <= tol
        midterm.termination_status = 1
        return #optimal
    end
    midterm.j = findfirst(x->x>tol,midterm.red_cost)

    d = zeros(length(c))
    d_base = B \ N[:,midterm.j]
    d[base] = - d_base
    d[nbase[midterm.j]] = 1
    midterm.d = d
    d_base = max.(d_base, 0)
    r = max.(xB, tol) ./ d_base
    val, midterm.i = findmin(r)
    
    if val == Inf
        midterm.termination_status = 2
        return #unbounded
    end
    midterm.z = c[base]'xB
    midterm.x = zeros(input.n)
    midterm.x[base] = xB
    iteration_log(input, midterm)
    base[midterm.i], nbase[midterm.j] = nbase[midterm.j], base[midterm.i] 
    return #max iteration
end