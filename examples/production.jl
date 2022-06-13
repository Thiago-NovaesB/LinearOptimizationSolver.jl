using LinearOptimizationSolver
using Plots
using Polyhedra

A = [2 1; 1 2]
b = [4, 4]
c = [4, 3]
input = create(A, b, c, tol=1E-10, solver = 1,verbose=false,standard=false)
output = solve(input)

x = output.X[1,:]
y = output.X[2,:]
n = length(x)
u = zeros(n)
v = zeros(n)
for i = 1:n-1
    u[i] = x[i+1] - x[i]
    v[i] = y[i+1] - y[i]
end

h = hrep([A; -1 0; 0 -1], [b; 0; 0])
p = polyhedron(h)

image = plot(p)
quiver!(x, y, quiver = (u, v))
quiver!([x[end]], [y[end]], quiver = ([c[1]]/4, [c[2]]/4))
plot!(x,y; seriestype = :scatter, title = "Problema da produção",legend = false)
savefig(image, "examples\\prod_path")

gap = output.gap

image = plot(log10.(gap),title = "Problema da produção",legend = false)
savefig(image, "examples\\prod_conv")


