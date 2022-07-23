"""
Sampling from and decoding an HMM
---------------------------------
This script shows how to sample points from a Hidden Markov Model (HMM):
we use a 4-state model with specified mean and covariance.
The plot shows the sequence of observations generated with the transitions
between them. We can see that, as specified by our transition matrix,
there are no transition between component 1 and 3.
Then, we decode our model to recover the input parameters.

Reference
https://github.com/hmmlearn/hmmlearn/blob/main/examples/plot_hmm_sampling_and_decoding.py
"""

# Silence WARNING:root:The use of `check_types` is deprecated and does not have any effect.
# https://github.com/tensorflow/probability/issues/1523
import logging

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from ssm_jax.hmm.learning import hmm_fit_em
from ssm_jax.hmm.models import GaussianHMM


class CheckTypesFilter(logging.Filter):

    def filter(self, record):
        return "check_types" not in record.getMessage()


def main(test_mode=False):
    logger = logging.getLogger()
    logger.addFilter(CheckTypesFilter())

    # Prepare parameters for a 4-components HMM
    # Initial population probability
    initial_probs = jnp.array([0.6, 0.3, 0.1, 0.0])
    # The transition matrix, note that there are no transitions possible
    # between component 1 and 3
    transition_matrix = jnp.array([[0.7, 0.2, 0.0, 0.1], [0.3, 0.5, 0.2, 0.0], [0.0, 0.3, 0.5, 0.2],
                                   [0.2, 0.0, 0.2, 0.6]])
    # The means of each component
    emission_means = jnp.array([[0.0, 0.0], [0.0, 11.0], [9.0, 10.0], [11.0, -1.0]])
    # The covariance of each component
    emission_covs = .5 * jnp.tile(jnp.identity(2), (4, 1, 1))

    # Build an HMM instance and set parameters
    gen_hmm = GaussianHMM(initial_probs, transition_matrix, emission_means, emission_covs)

    # Generate samples
    key = jr.PRNGKey(42)
    num_timesteps = 500
    states, emissions = gen_hmm.sample(key, num_timesteps)

    if not test_mode:
        # Plot the sampled data
        fig, ax = plt.subplots()
        ax.plot(emissions[:, 0], emissions[:, 1], ".-", label="observations", ms=6, mfc="orange", alpha=0.7)

        # Indicate the component numbers
        for i, m in enumerate(emission_means):
            ax.text(m[0],
                    m[1],
                    'Component %i' % (i + 1),
                    size=17,
                    horizontalalignment='center',
                    bbox=dict(alpha=.7, facecolor='w'))
        ax.legend(loc='best')
        fig.show()
        plt.savefig("gaussian-hmm-sampled-data.png")

    # %%
    # Now, let's ensure we can recover our parameters.

    scores = list()
    models = list()
    for n_components in (3, 4, 5):
        for idx in range(10):
            # define our hidden Markov model
            hmm = GaussianHMM.random_initialization(jr.PRNGKey(idx), n_components, emissions.shape[-1])
            # 50/50 train/validate
            hmm, *_ = hmm_fit_em(hmm, emissions[:emissions.shape[0] // 2][None, ...])
            models.append(hmm)
            scores.append(hmm.marginal_log_prob(emissions[emissions.shape[0] // 2:]))
            print(f'\tScore: {scores[-1]}')

    if not test_mode:
        # get the best model
        hmm = models[jnp.argmax(jnp.array(scores))]
        n_states = hmm.num_states
        print(f'The best model had a score of {max(scores)} and {n_states} '
              'states')

        # use the Viterbi algorithm to predict the most likely sequence of states
        # given the model
        most_likely_states = hmm.most_likely_states(emissions[None, ...])

        # %%
        # Let's plot our states compared to those generated and our transition matrix
        # to get a sense of our model. We can see that the recovered states follow
        # the same path as the generated states, just with the identities of the
        # states transposed (i.e. instead of following a square as in the first
        # figure, the nodes are switch around but this does not change the basic
        # pattern). The same is true for the transition matrix.

        # plot model states over time
        fig, ax = plt.subplots()
        ax.plot(states, most_likely_states)
        ax.set_title('States compared to generated')
        ax.set_xlabel('Generated State')
        ax.set_ylabel('Recovered State')
        fig.show()
        plt.savefig("gaussian-hmm-most-likely-states.png")

        # plot the transition matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.imshow(gen_hmm.transition_matrix, aspect='auto', cmap='spring')
        ax1.set_title('Generated Transition Matrix')
        ax2.imshow(hmm.transition_matrix, aspect='auto', cmap='spring')
        ax2.set_title('Recovered Transition Matrix')
        for ax in (ax1, ax2):
            ax.set_xlabel('State To')
            ax.set_ylabel('State From')

        fig.tight_layout()
        fig.show()
        plt.savefig("gaussian-hmm-transition-matrix.png")


if __name__ == "__main__":
    main()
