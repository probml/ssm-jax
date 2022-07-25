from jax import random as  jr
from jax import numpy as jnp
from jax import lax

import os
os.chdir('/home/xinglong/git_local/ssm-jax/ssm_jax/lgssm')
from blocked_gibbs import lgssm_blocked_gibbs
from models import LinearGaussianSSM


def test_lgssm_map(num_itrs=100, timesteps=100, seed=jr.PRNGKey(0)):
    
    # Set the dimension of the system
    dim_obs = 2
    dim_hid = 4
    dim_in = 6

    # Set true value of the parameter
    Q = 1e-3 * jnp.eye(dim_hid)
    delta = 1.
    F = jnp.array([[1, 0, delta, 0], [0, 1, 0, delta], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = jnp.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]])
    R = 1e-3 * jnp.eye(dim_obs)
    H = jnp.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    D = jnp.array([[0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
    
    # Set the input
    u1, u2 = 1., 2.
    inputs = jnp.tile(jnp.array([u1, u1, u1, u1, u2, u2]), (timesteps, 1))
    
    # Generate the observation
    key = iter(jr.split(seed, 4))
    initial_state = jr.multivariate_normal(next(key), jnp.ones(dim_hid), Q)
    noise_dynamics = jr.multivariate_normal(next(key), jnp.zeros(dim_hid), Q, shape=(timesteps-1, ))
    noise_emission = jr.multivariate_normal(next(key), jnp.zeros(dim_obs), R, shape=(timesteps, ))
    
    def state_update(state, extras):
        input, noise_dyn, noise_ems = extras
        state_new = F.dot(state)  + B.dot(input) + noise_dyn
        emission = H.dot(state_new) + D.dot(input) + noise_ems
        return state_new, emission
    
    emission_1st = H.dot(initial_state) + D.dot(inputs[0]) + noise_emission[0]
    _, emissions = lax.scan(state_update, initial_state, (inputs[1:], noise_dynamics, noise_emission[1:])) 
    emissions = jnp.row_stack((emission_1st, emissions))   
    
    # Set the hyperparameter for the prior distribution of parameters
    mu_init, Cov_init = jnp.zeros(dim_hid), jnp.eye(dim_hid)
    M_dyn, V_dyn, nu_dyn, Psi_dyn = jnp.zeros((dim_hid, dim_hid+dim_in)), jnp.eye(dim_hid+dim_in), dim_hid, jnp.eye(dim_hid)
    M_ems, V_ems, nu_ems, Psi_ems = jnp.zeros((dim_obs, dim_hid+dim_in)), jnp.eye(dim_hid+dim_in), dim_obs, jnp.eye(dim_obs)
    initial_prior_params = (mu_init, Cov_init)
    dynamics_prior_params = (M_dyn, V_dyn, nu_dyn, Psi_dyn)
    emission_prior_params = (M_ems, V_ems, nu_ems, Psi_ems)
    prior_hyperparams = (initial_prior_params, dynamics_prior_params, emission_prior_params)

    