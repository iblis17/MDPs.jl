
sprand_reward(::Type{R}, states, actions) where {R} =
    sprand(states, actions, 1/3, x -> rand_reward(R, x))


import Base.SparseMatrix.sprand_IJ


function sprand_transition(::Type{T}, states) where T
    I, J = VERSION < v"0.4-" ?
        sprand_IJ(states, states, 1/3) :
        sprand_IJ(Base.Random.GLOBAL_RNG, states, states, 1/3)
    V = rand(T, length(I))
    for j = 1:states
        idx = find(J .== j)
        if length(idx) == 0
            i = rand(1:states)
            push!(I, i)
            push!(J, j)
            push!(V, one(T))
        else
            V[idx] = V[idx] ./ sum(V[idx])
        end
    end
    sparse(I, J, V, states, states)
end


function sprandom(
    ::Type{P},
    ::Type{R},
    states,
    actions,
) where {P<:FloatingPoint,R<:FloatingPoint}
    transition = SparseMatrixCSC{P,Int}[
        sprand_transition(P, states) for a = 1:actions
    ]
    reward = sprand_reward(R, states, actions)
    (transition, reward)
end


sprandom(states, actions) = sprandom(Float64, Float64, states, actions)
