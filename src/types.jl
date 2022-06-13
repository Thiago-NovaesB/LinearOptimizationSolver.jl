mutable struct Input
    A::Matrix{}
    b::Vector{}
    c::Vector{}
    standard::Bool 
    n::Int
    m::Int
    rho::Float64
    alpha::Float64
    tol::Float64
    max_iter::Int
    verbose::Bool
    solver::Int
    crossover::Bool
    crossover_tol::Float64
end

Base.@kwdef mutable struct MidTerm
    termination_status::Int = 0
    iter::Int = 0
    d::Vector{} = []
    base::Vector{Int} = []
    nbase::Vector{Int} = []
    i::Int = 0
    j::Int = 0
    z::Float64 = 0.0
    x::Vector{Float64} = []
    red_cost::Vector{Float64} = []
end

struct OutputIP
    x::Vector{}
    s::Vector{}
    p::Vector{}
    X::Matrix{}
    S::Matrix{}
    P::Matrix{}
    gap::Vector{}
    mu::Float64
    z::Float64
    termination_status::Int
    iter::Integer
end

struct OutputSimplex
    x::Vector{Float64}
    z::Float64
    termination_status::Int
    base::Vector{Int}
    nbase::Vector{Int}
end

