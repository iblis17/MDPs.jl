import Base: convert, ndims, size



# --------------
# Abstract types
# --------------

@doc """
# Interface

* probability(P, s, t, a) -> the probability of transitioning from state s to state t given action a

""" ->
abstract type AbstractTransitionProbability end


@doc """
# Interface

* probability(P, a) -> probability transition matrix for action a

## Optional

* getindex(P, s, t, a) -> the probability of transitioning from state s to state t given action a

""" ->
abstract type AbstractArrayTransitionProbability <: AbstractTransitionProbability end



# ----------
# Array type
# ----------

struct ArrayTransitionProbability{T<:Real} <: AbstractArrayTransitionProbability
    array::Array{T,3}

    function ArrayTransitionProbability(array::Array{T,3})
        is_square_stochastic(array) || error("Not valid transition probability.")
        new(array)
    end
end

ArrayTransitionProbability(array::Array{T,3}) where {T} = ArrayTransitionProbability{T}(array)


TransitionProbability(A::Array{T,3}) where {T} = ArrayTransitionProbability(A)


# Base
# ----

ndims(P::ArrayTransitionProbability) = ndims(P.array)

size(P::ArrayTransitionProbability) = size(P.array)
size(P::ArrayTransitionProbability, dims...) = size(P.array, dims...)

getindex(P::ArrayTransitionProbability, dims...) = getindex(P.array, dims...)


# num
# ---

num_actions(P::ArrayTransitionProbability) = size(P, 3)

num_states(P::ArrayTransitionProbability) = size(P, 1)


# probability
# -----------

@doc """
# Transition probability

The probability of transitioning from state `s` to
state `t` given that action `a` is taken.

""" ->
probability(P::ArrayTransitionProbability, s, t, a) = getindex(P, s, t, a)

probability(P::ArrayTransitionProbability, a) = P[:, :, a]



# -----------------
# Sparse array type
# -----------------

struct SparseArrayTransitionProbability{Tv<:Real,Ti} <: AbstractArrayTransitionProbability
    array::Vector{SparseMatrixCSC{Tv,Ti}}
end


TransitionProbability(A::Vector{T}) where {T<:SparseMatrixCSC} =
    SparseArrayTransitionProbability(A)


num_actions(P::SparseArrayTransitionProbability) = length(P.array)

num_states(P::SparseArrayTransitionProbability) = size(getindex(P.array, 1), 1)


probability(P::SparseArrayTransitionProbability, s, t, a) =
    getindex(getindex(P.array, a), s, t)

probability(P::SparseArrayTransitionProbability, a) = getindex(P.array, a)



# -------------
# Function type
# -------------

struct FunctionTransitionProbability <: AbstractTransitionProbability
    func::Function
    states::Int
    actions::Int
end


TransitionProbability(F::Function, states, actions) = FunctionTransitionProbability(F, states, actions)


num_actions(P::FunctionTransitionProbability) = P.actions

num_states(P::FunctionTransitionProbability) = P.states


probability(P::FunctionTransitionProbability, s, t, a) = P.func(s, t, a)
