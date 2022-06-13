function solve_ip(input::Input)

    init_log_ip(input)
    n = input.n
    m = input.m
    rho = input.rho
    alpha = input.alpha
    tol = input.tol
    A = input.A
    b = input.b
    c = input.c

    nx = (1:n)
    np = (n+1:m+n)
    ns = (m+n+1:m+2n)

    x = ones(n)
    s = ones(n)
    p = ones(m)
    X = x
    S = s
    P = p
    gap = []
    mu = rho*x'*s/n

    for iter in 1:input.max_iter
        F = [A zeros(m,m) zeros(m,n);
            zeros(n,n) A' I(n);
            Diagonal(s) zeros(n,m) Diagonal(x)]

        g = [A*x - b; A'p + s + c; x.*s .- mu]
        
        d = - F \ g

        d_x = d[nx]
        d_p = d[np]
        d_s = d[ns]

        filter_x = d_x .< 0.0
        filter_p = d_p .< 0.0
        filter_s = d_s .< 0.0

        if sum(filter_x) == 0
            beta_x = 1.0
        else
            beta_x = min(1.0, alpha*minimum((-x./d_x)[filter_x]))
        end 

        if sum(filter_p) == 0
            beta_p = 1.0
        else
            beta_p = min(1.0, alpha*minimum((-p./d_p)[filter_p]))
        end
        
        if sum(filter_s) == 0
            beta_s = 1.0
        else
            beta_s = min(1.0, alpha*minimum((-s./d_s)[filter_s]))
        end

        x += beta_x * d_x
        p += beta_s * d_p
        s += beta_s * d_s
        z = c'x
        w = b'p
        dual_inf = x'*s

        if z > 1/tol
            output = OutputIP(x,s,p,X,S,P,gap,mu,Inf,2,input.max_iter)
            last_log(input, output)
            return output
        end

        if w > 1/tol
            output = OutputIP(x,s,p,X,S,P,gap,mu,-Inf,3,input.max_iter)
            last_log(input, output)
            return output
        end
        X = [X x]
        S = [S s]
        P = [P p]
        push!(gap,dual_inf)
        iteration_log(input,iter,z,dual_inf)
        mu = rho*dual_inf/n
        if mu < tol
            output = OutputIP(x,s,p,X,S,P,gap,mu,z,1,iter)
            last_log(input, output)
            return output
        end
    end
    z = c'x
    output = OutputIP(x,s,p,X,S,P,gap,mu,z,0,input.max_iter)
    last_log(input, output)
    return output
end

function crossover(input::Input, output::OutputIP)
    m = input.m
    n = input.n
    crossover_tol = input.crossover_tol
    x = output.x
    midterm = MidTerm()

    base = []
    nbase = []

    for i in 1:n
        if x[i] > 1E-3
            push!(base,i)
        else
            push!(nbase,i)
        end
    end

    midterm.base = base
    midterm.nbase = nbase

    B = view(input.A,:,midterm.base)
    for (k,w) in enumerate(midterm.nbase)
        sol = B \ input.A[:,w]
        if norm(B*sol - input.A[:,w]) > crossover_tol
            push!(midterm.base, w)
            deleteat!(midterm.nbase, k)
            if length(midterm.base) == m
                break
            end
        end
    end

    init_log_simplex2(input)
    while midterm.termination_status == 0 && midterm.iter < input.max_iter
        iterate!(input, midterm)
    end
    update_midterm!(input, midterm)
    output = write_output(input, midterm)
    return output
end