from jax import numpy as jnp

def block_split(A, idx):
    """Split square matrix A into four blocks at `idx`
    
    For example splitting
        [[0, 1, 2],
         [3, 4, 5],
         [6, 7 ,8]]
    at idx = 2 produces:
        ([[0,1],[3,4]],
         [[2],[5]],
         [[6,7]],
         [[8]]).

    Args:
        A (D, D): array.
        idx: int.
    Returns:
        A11 (dim1, dim1)
        A12 (dim2, dim1)
        A21 (dim1, dim2)
        A22 (dim2, dim2)
       where `dim1 + dim2 == D` and `dim2 = D - idx`.
    """
    split_array = jnp.array([idx])
    vblocks = jnp.vsplit(A,split_array)
    # [leaf for tree in forest for leaf in tree]
    blocks = [block for vblock in vblocks for block in jnp.hsplit(vblock,split_array)]
    # Can also do:
    #   blocks = tree_map(lambda arr: jnp.hsplit(arr,split_array),vblocks)
    # followed by a tree_flatten
    return blocks

def info_marginalise(K, h, idx):
    """Calculate the parameters of marginalised MVN.
    
    For x1, x2 joint distributed as
        p(x1, x2) = Nc(x1,x2| h, K),
    the marginal distribution of x1 is given by:
        p(x1) = \int p(x1, x2) dx2 = Nc(x1 | h1_marg, K1_marg)
    where,
        h1_marg = h2 - K21 K11^{-1} h1
        K1_marg = K22 - K21 K11^{-1} K12
        h = (h1 h2)^T
        K = (K11; K12, K21; K22)

    Args:
        K (D, D): joint precision matrix.
        h (D,1): joint precision weighted mean.
        idx: the index which divides the joint x into (x1,x2),
              equal to `len(x1) + 1`.
    Returns:
        K1_marg (dim1, dim1): marginal precision matrix.
        h1_marg (dim1,1): marginal precision weighted mean.
    """
    K11, K12, K21, K22 = block_split(K, idx)
    h1 = h[:idx]
    h2 = h[idx:]
    G = jnp.linalg.solve(K11,K12)
    K1_marg = K22 - K21 @ G
    h1_marg = h2 - G.T @ h1
    return K1_marg, h1_marg

def info_condition(K, h, y):
    """Calculate the parameters of MVN after conditioning."""
    idx = len(h) - len(y)
    K11, K12, *_ = block_split(K, idx)
    h1 = h[:idx]
    return K11, h1 - K12 @ y

def potential_from_conditional_linear_gaussian(A,u,Lambda):
    """Express a conditional linear gaussian as a potential in canonical form.
    
    p(y|z) = N(y | Az + u, Lambda^{-1})
           \prop exp( -0.5(y z)^T K (y z) + (y z)^T h )
    where,
        K = (Lambda; -Lambda A,  -A.T Lambda; A.T Lambda A)
        h = (Lambda u, -A.T Lambda u)

    Args:
        A (dim1, dim2)
        u (dim1, 1)
        Lambda (dim1, dim1)
    Returns:
        K (dim1 + dim2, dim1 + dim2)
        h (dim1 + dim2, 1)
    """
    dim_y, _ = A.shape
    I = jnp.eye(dim_y)
    IA = jnp.vstack((I, -A.T))
    K = IA @ Lambda @ IA.T
    Lu  = Lambda @ u
    h = jnp.vstack((Lu, -A.T @ Lu))
    return K, h


def info_multiply(params1, params2):
    """Calculate parameters resulting from multiplying gaussians."""
    return jax.tree_map(lambda a,b: a + b, params1, params2)


def info_divide(params1, params2):
    """Calculate parameters resulting from dividing gaussians."""
    return jax.tree_map(lambda a,b: a - b, params1, params2)

