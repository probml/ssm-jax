import jax.numpy as jnp
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import nn
from jax import vmap
from jax.tree_util import register_pytree_node_class
from ssm_jax.hmm.models.base import BaseHMM

# Using TFP for now since it has all our distributions
# (Distrax doesn't have Poisson, it seems.)


@register_pytree_node_class
class PoissonHMM(BaseHMM):

    def __init__(self, initial_probabilities, transition_matrix, emission_log_rates):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_rates (_type_): _description_
        """
        super().__init__(initial_probabilities, transition_matrix)
        self._emission_distribution = tfd.Poisson(log_rate=emission_log_rates)

    @classmethod
    def random_initialization(cls, key, num_states, emission_dim):
        key1, key2, key3 = jr.split(key, 3)
        initial_probs = jr.dirichlet(key1, jnp.ones(num_states))
        transition_matrix = jr.dirichlet(key2, jnp.ones(num_states), (num_states,))
        emission_log_rates = jnp.log(jr.exponential(key3, (num_states, emission_dim)))
        return cls(initial_probs, transition_matrix, emission_log_rates)

    # Properties to get various parameters of the model
    @property
    def num_obs(self):
        return self.emission_log_rates.shape[-1]

    @property
    def emission_rates(self):
        return jnp.exp(self.emission_log_rates)

    @property
    def emission_log_rates(self):
        return self._emission_distribution.log_rate

    @property
    def unconstrained_params(self):
        """Helper property to get a PyTree of unconstrained parameters.
        """
        return (nn.softmax(jnp.log(self.initial_probabilities),
                           axis=-1), nn.softmax(jnp.log(self.transition_matrix), axis=-1), self.emission_log_rates)

    @classmethod
    def from_unconstrained_params(cls, unconstrained_params, hypers):
        return cls(*unconstrained_params, *hypers)

    def m_step(self, batch_emissions, batch_posteriors, batch_trans_probs, optimizer=optax.adam(0.01), num_iters=50):
        # Based on: Hyvönen & Tolonen, "Bayesian Inference 2019"
        # section 3.2
        # https://vioshyvo.github.io/Bayesian_inference
        smoothed_probs = batch_posteriors.smoothed_probs
        print(smoothed_probs.shape, batch_emissions.shape)
        one_hot_emissions = nn.one_hot(jnp.squeeze(batch_emissions), num_classes=self.num_obs, axis=-1)
        print(one_hot_emissions.shape)
        obs_probs = vmap(lambda x, y: jnp.dot(x.T, y))(smoothed_probs, one_hot_emissions)
        obs_probs = jnp.sum(obs_probs, axis=0)
        posterior_sum = smoothed_probs.sum(axis=1).sum(axis=0)

        n = posterior_sum.sum()
        y_bar = obs_probs / posterior_sum
        emission_log_rates = jnp.log((n * y_bar) / n)

        transitions_probs = batch_trans_probs.sum(axis=0)
        denom = transitions_probs.sum(axis=-1, keepdims=True)
        transitions_probs = transitions_probs / jnp.where(denom == 0, 1, denom)

        batch_initial_probs = smoothed_probs[:, 0, :]
        initial_probs = batch_initial_probs.sum(axis=0) / batch_initial_probs.sum()

        hmm = PoissonHMM(initial_probs, transitions_probs, emission_log_rates)

        return hmm, batch_posteriors.marginal_loglik
