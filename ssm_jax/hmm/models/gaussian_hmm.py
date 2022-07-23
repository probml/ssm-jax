from dataclasses import replace
from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_map
from ssm_jax.hmm.inference import compute_transition_probs
from ssm_jax.hmm.inference import hmm_smoother
from ssm_jax.hmm.models.base import BaseHMM
from ssm_jax.hmm.models.base import Parameter
from ssm_jax.utils import PSDToRealBijector


@register_pytree_node_class
class GaussianHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_means, emission_covariance_matrices):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_means (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)

        self._emission_param = Parameter((emission_means, emission_covariance_matrices))

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_means = jr.normal(key3, (num_states, emission_dim))
        emission_covs = jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1))
        return cls(initial_probs, transition_matrix, emission_means, emission_covs)

    # Properties to get various parameters of the model
    @property
    def emission_means(self):
        return self._emission_param.value[0]

    @property
    def emission_covariance_matrices(self):
        return self._emission_param.value[1]

    def emission_distribution(self, state):
        emission_means, emission_covs = self._emission_param.value
        return tfd.MultivariateNormalFullCovariance(emission_means[state], emission_covs[state])

    # Expectation-maximization (EM) code
    def e_step(self, batch_emissions):
        """The E-step computes expected sufficient statistics under the
        posterior. In the Gaussian case, this these are the first two
        moments of the data
        """

        @chex.dataclass
        class GaussianHMMSuffStats:
            # Wrapper for sufficient statistics of a GaussianHMM
            marginal_loglik: chex.Scalar
            initial_probs: chex.Array
            trans_probs: chex.Array
            sum_w: chex.Array
            sum_x: chex.Array
            sum_xxT: chex.Array

        def _single_e_step(emissions):
            # Run the smoother
            posterior = hmm_smoother(self.initial_probabilities, self.transition_matrix,
                                     self._conditional_logliks(emissions))

            # Compute the initial state and transition probabilities
            initial_probs = posterior.smoothed_probs[0]
            trans_probs = compute_transition_probs(self.transition_matrix, posterior)

            # Compute the expected sufficient statistics
            sum_w = jnp.einsum("tk->k", posterior.smoothed_probs)
            sum_x = jnp.einsum("tk, ti->ki", posterior.smoothed_probs, emissions)
            sum_xxT = jnp.einsum("tk, ti, tj->kij", posterior.smoothed_probs, emissions, emissions)

            # TODO: might need to normalize x_sum and xxT_sum for numerical stability
            stats = GaussianHMMSuffStats(marginal_loglik=posterior.marginal_loglik,
                                         initial_probs=initial_probs,
                                         trans_probs=trans_probs,
                                         sum_w=sum_w,
                                         sum_x=sum_x,
                                         sum_xxT=sum_xxT)
            return stats

        # Map the E step calculations over batches
        return vmap(_single_e_step)(batch_emissions)

    def freeze_emission_params(self):
        self._emission_param = replace(self._emission_param,
                                       is_trainable=False)
        
    def m_step(self, batch_emissions, batch_posteriors, **kwargs):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_posteriors)

        # Initial distribution
        self._initial_probabilities = replace(self._initial_probabilities,
                                              value=tfd.Dirichlet(1.0001 + stats.initial_probs).mode())

        # Transition distribution
        self._transition_matrix = replace(self._transition_matrix,
                                          value=tfd.Dirichlet(1.0001 + stats.trans_probs).mode())

        # Gaussian emission distribution
        emission_dim = stats.sum_x.shape[-1]
        emission_means = stats.sum_x / stats.sum_w[:, None]
        _emission_means = self._emission_param.value[0]
        emission_covs = (stats.sum_xxT / stats.sum_w[:, None, None] -
                         jnp.einsum("ki,kj->kij", _emission_means, _emission_means) + 1e-4 * jnp.eye(emission_dim))
        self._emission_param = Parameter((emission_means, emission_covs))

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters."""
        return (
            tfb.SoftmaxCentered().inverse(self.initial_probabilities),
            tfb.SoftmaxCentered().inverse(self.transition_matrix),
            self.emission_means,
            PSDToRealBijector.forward(self.emission_covariance_matrices),
        )

    @unconstrained_params.setter
    def unconstrained_params(self, unconstrained_params):
        self._initial_probabilities = replace(self._initial_probabilities,
                                              value=tfb.SoftmaxCentered().forward(unconstrained_params[0]))
        self._transition_matrix = replace(self._transition_matrix,
                                          value=tfb.SoftmaxCentered().forward(unconstrained_params[1]))
        self._emission_param = Parameter((unconstrained_params[2], PSDToRealBijector.inverse(unconstrained_params[3])))
