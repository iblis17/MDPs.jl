# --------------
# Abstract types
# --------------

@doc """
# Interface

* reward(R, s, a)

## Optional

* reward(R, s, s, a)
* reward(R, s)

""" ->
abstract type AbstractReward end

abstract type AbstractArrayReward <: AbstractReward end


getindex(R::AbstractArrayReward, dims...) = getindex(R.array, dims...)

size(R::AbstractArrayReward, d) = size(R.array, d)


num_states(R::AbstractArrayReward) = size(R, 1)


# ----------
# Array type
# ----------

struct ArrayReward{T<:Real,N} <: AbstractArrayReward
    array::Array{T,N}

    function ArrayReward{T,N}(array::Array{T,N}) where {T,N}
        N < 3 || (N == 3 && size(array, 1) == size(array, 2)) ||
            error("ArrayReward can not constructed with this Array.")
        new(array)
    end
end

ArrayReward(array::Array{T,N}) where {T,N} = ArrayReward{T,N}(array)


# type construction helper
Reward(A::Array) = ArrayReward(A)


num_actions(R::ArrayReward{T,1}) where {T} =
    error(string(typeof(R))*" does not know about actions.")
num_actions(R::ArrayReward{T,N}) where {T,N} = size(R, N)


# 1 dimension
# -----------

reward(R::ArrayReward{T,1}, state) where {T} = getindex(R, state)

reward(R::ArrayReward{T,1}, state, action) where {T} = reward(R, state)

reward(R::ArrayReward{T,1}, state, new_state, action) where {T} = reward(R, state)


# 2 dimensions
# ------------

reward(R::ArrayReward{T,2}, state, action) where {T} = getindex(R, state, action)

reward(R::ArrayReward{T,2}, state, new_state, action) where {T} = reward(R, state, action)

# 3 dimensions
# ------------

reward(R::ArrayReward{T,3}, state, new_state, action) where {T} =
    getindex(R, state, new_state, action)



# -----------
# Sparse type
# -----------

struct SparseReward{Tv<:Real,Ti} <: AbstractArrayReward
    array::SparseMatrixCSC{Tv,Ti}
end


Reward(A::SparseMatrixCSC) = SparseReward(A)


num_actions(R::SparseReward) = size(R, 2)


reward(R::SparseReward, state, action) = getindex(R, state, action)

reward(R::SparseReward, state, new_state, action) = reward(R, state, action)
