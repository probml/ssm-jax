import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import nn
from jax import vmap
from jax.tree_util import register_pytree_node_class
from ssm_jax.hmm.models.base import BaseHMM


@register_pytree_node_class
class MultinomialHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_probs, num_trials=1):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_probs (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        self._num_trials = num_trials
        self._emission_distribution = tfd.Multinomial(num_trials, probs=emission_probs)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_probs = jr.uniform(key3, (num_states, emission_dim))
        return cls(initial_probs, transition_matrix, emission_probs)

    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return (tfb.SoftmaxCentered().inverse(self.initial_probabilities),
                tfb.SoftmaxCentered().inverse(self.transition_matrix), tfb.Sigmoid().inverse(self.emission_probs))

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        initial_probabilities = tfb.SoftmaxCentered().forward(unconstrained_params[0])
        transition_matrix = tfb.SoftmaxCentered().forward(unconstrained_params[1])
        emission_probs = tfb.Sigmoid().forward(unconstrained_params[2])
        return cls(initial_probabilities, transition_matrix, emission_probs, *hypers)

    @property
    def num_trials(self):
        return self._num_trials

    def m_step(self, batch_emissions, batch_posteriors, batch_trans_probs, optimizer=optax.adam(0.01), num_iters=50):

        smoothed_probs = batch_posteriors.smoothed_probs
        """state_probs = smoothed_probs.sum(axis=1)
        denom = state_probs.sum(axis=-1, keepdims=True)
        state_probs = state_probs / jnp.where(denom == 0, 1, denom)
        counts = vmap(lambda x: jnp.bincount(x.astype(jnp.int32), length=self.num_obs))(batch_emissions)
        emission_probs = state_probs.T @ counts
        denom = emission_probs.sum(axis=-1, keepdims=True)
        emission_probs = emission_probs / jnp.where(denom == 0, 1, denom)"""

        one_hot_emissions = nn.one_hot(batch_emissions, num_classes=self.num_obs, axis=-1)
        emission_probs = vmap(lambda x, y: jnp.dot(x.T, y))(smoothed_probs, one_hot_emissions)
        emission_probs = jnp.sum(emission_probs, axis=0)
        denom = emission_probs.sum(axis=-1, keepdims=True)
        emission_probs = emission_probs / jnp.where(denom == 0, 1, denom)

        transitions_probs = batch_trans_probs.sum(axis=0)
        denom = transitions_probs.sum(axis=-1, keepdims=True)
        transitions_probs = transitions_probs / jnp.where(denom == 0, 1, denom)

        batch_initial_probs = smoothed_probs[:, 0, :]
        initial_probs = batch_initial_probs.sum(axis=0) / batch_initial_probs.sum()

        hmm = MultinomialHMM(initial_probs, transitions_probs, emission_probs, self._num_trials)

        return hmm, batch_posteriors.marginal_loglik
