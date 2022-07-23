from abc import ABC
from abc import abstractmethod
from dataclasses import replace
from functools import partial
from typing import Optional

import chex
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import lax
from jax import vmap
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_filter
from ssm_jax.hmm.inference import hmm_posterior_mode
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.inference import hmm_two_filter_smoother
from ssm_jax.hmm.learning import hmm_fit_sgd


class Parameter(eqx.Module):
    value: chex.ArrayTree
    is_trainable: Optional[bool] = eqx.static_field()
    to_unconstrained: Optional[tfd.Distribution] = eqx.static_field()

    def __init__(self, value, is_trainable=True, to_unconstrained=None):
        super().__init__()
        self.value = value
        self.is_trainable = is_trainable
        self.to_unconstrained = to_unconstrained


class BaseHMM(ABC):

    def __init__(self, initial_probabilities, transition_matrix):
        """
        Abstract base class for Hidden Markov Models.
        Child class specifies the emission distribution.

        Args:
            initial_probabilities[k]: prob(hidden(1)=k)
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        # Check shapes
        num_states = transition_matrix.shape[-1]
        assert initial_probabilities.shape == (num_states,)
        assert transition_matrix.shape == (num_states, num_states)

        # Store the parameters
        self._initial_probabilities = Parameter(initial_probabilities)
        self._transition_matrix = Parameter(transition_matrix)

    # Properties to allow unconstrained optimization
    @property
    @abstractmethod
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        raise NotImplementedError

    @unconstrained_params.setter
    @abstractmethod
    def unconstrained_params(self, value):
        raise NotImplementedError

    @property
    def hyperparams(self):
        """Helper property to get a PyTree of model hyperparameters."""
        return None

    @hyperparams.setter
    def hyperparams(self, value):
        pass

    # Properties to get various attributes of the model.
    @property
    def num_states(self):
        return self.initial_distribution().probs_parameter().shape[0]

    @property
    def num_obs(self):
        return self.emission_distribution(0).event_shape[0]

    def initial_distribution(self):
        return tfd.Categorical(probs=self._initial_probabilities.value)

    def transition_distribution(self, state):
        return tfd.Categorical(probs=self._transition_matrix.value[state])

    @property
    def initial_probabilities(self):
        return self.initial_distribution().probs_parameter()

    @property
    def transition_matrix(self):
        # Note: This will generalize to models with transition *functions*
        return vmap(lambda state: \
            self.transition_distribution(state).probs_parameter())(
            jnp.arange(self.num_states))

    def freeze_transition_matrix(self):
        self._transition_matrix = replace(self._transition_matrix, is_trainable=False)

    def freeze_initial_probabilities(self):
        self._initial_probabilities = replace(self._initial_probabilities, is_trainable=False)

    @abstractmethod
    def emission_distribution(self, state):
        raise NotImplementedError

    def sample(self, key, num_timesteps):
        """Sample a sequence of latent states and emissions.

        Args:
            key: rng key
            num_timesteps: length of sequence to generate
        """

        def _step(state, key):
            key1, key2 = jr.split(key, 2)
            emission = self.emission_distribution(state).sample(seed=key1)
            next_state = self.transition_distribution(state).sample(seed=key2)
            return next_state, (state, emission)

        # Sample the initial state
        key1, key = jr.split(key, 2)
        initial_state = self.initial_distribution().sample(seed=key1)

        # Sample the remaining emissions and states
        keys = jr.split(key, num_timesteps)
        _, (states, emissions) = lax.scan(_step, initial_state, keys)
        return states, emissions

    def log_prob(self, states, emissions):
        """Compute the log joint probability of the states and observations"""
        lp = self.initial_distribution().log_prob(states[0])
        lp += self.transition_distribution(states[:-1]).log_prob(states[1:]).sum()
        f = lambda state, emission: \
            self.emission_distribution(state).log_prob(emission)
        lp += vmap(f)(states, emissions).sum()
        return lp

    def _conditional_logliks(self, emissions):
        # Compute the log probability for each time step by
        # performing a nested vmap over emission time steps and states.
        f = lambda emission: \
            vmap(lambda state: \
                self.emission_distribution(state).log_prob(emission))(
                    jnp.arange(self.num_states)
                )
        return vmap(f)(emissions)

    # Basic inference code
    def marginal_log_prob(self, emissions):
        """Compute log marginal likelihood of observations."""
        post = hmm_filter(self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions))
        ll = post.marginal_loglik
        return ll

    def most_likely_states(self, emissions):
        """Compute Viterbi path."""
        return hmm_posterior_mode(self.initial_probabilities, self.transition_matrix,
                                  self._conditional_logliks(emissions))

    def filter(self, emissions):
        """Compute filtering distribution."""
        return hmm_filter(self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions))

    def smoother(self, emissions):
        """Compute smoothing distribution."""
        return hmm_smoother(self.initial_probabilities, self.transition_matrix, self._conditional_logliks(emissions))

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the generic case, we simply return the posterior itself.
        """

        def _single_e_step(emissions):
            # TODO: do we need to use dynamic slice?
            posterior = hmm_two_filter_smoother(self.initial_probabilities, self.transition_matrix,
                                                self._conditional_logliks(emissions))

            # Compute the transition probabilities
            posterior.trans_probs = compute_transition_probs(self.transition_matrix, posterior)

            return posterior

        return vmap(_single_e_step)(batch_emissions)

    def m_step(self, batch_emissions, batch_posteriors, optimizer=optax.adam(1e-2), num_mstep_iters=50):
        """_summary_

        Args:
            emissions (_type_): _description_
            posterior (_type_): _description_
        """
        hypers = self.hyperparams

        def _single_expected_log_joint(hmm, emissions, posterior, trans_probs):
            # TODO: This needs help! Ideally, posterior would include trans_probs as a field
            posterior, trans_probs = posterior

            # TODO: do we need to use dynamic slice?
            log_likelihoods = hmm._conditional_logliks(emissions)
            expected_states = posterior.smoothed_probs

            lp = jnp.sum(expected_states[0] * jnp.log(hmm.initial_probabilities))
            lp += jnp.sum(trans_probs * jnp.log(hmm.transition_matrix))
            lp += jnp.sum(expected_states * log_likelihoods)
            return lp

        def neg_expected_log_joint(params):
            hmm = self.from_unconstrained_params(params, hypers)
            f = vmap(partial(_single_expected_log_joint, hmm))
            lps = f(batch_emissions, batch_posteriors)
            return -jnp.sum(lps / jnp.ones_like(batch_emissions).sum())

        # TODO: minimize the negative expected log joint with SGD

        hmm, losses = hmm_fit_sgd(self,
                                  batch_emissions,
                                  optimizer=optimizer,
                                  num_iters=num_mstep_iters,
                                  loss_fn=neg_expected_log_joint)
        return hmm

    # Use the to/from unconstrained properties to implement JAX tree_flatten/unflatten
    def tree_flatten(self):
        children = self.unconstrained_params
        aux_data = self.hyperparams
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.unconstrained_params = children
        obj.hyperparams = aux_data
        return obj
