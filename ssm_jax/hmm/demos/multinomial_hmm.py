"""
A simple example demonstrating Multinomial HMM
Based on an example here: https://github.com/hmmlearn/hmmlearn/blob/main/examples/multinomial_hmm_example.py
... which was implementing a Categorical HMM
Multinomial HMM is a generalization of Categorical HMM, key differences being:
    - a Categorical (aka generalized Bernoulli/multinoulli) distribution models
    an outcome of a die with `n_features` possible values, i.e. it is a
    generaliztion of the Bernoulli distribution where there are `n_features`
    categories instead of the binary success/failure outcome;
    a Categorical HMM has the emission probabilities for each component
    parametrized by Categorical distributions
    - a Multinomial distribution models the outcome of `n_trials` independent
    rolls of die, each with `n_features` possible values; i.e.
      - when n_trials = 1 and n_features = 1, Multinomial is the
        Bernoulli distribution
      - when n_trials > 1 and n_features = 2, Multinomial is the
        Binomial distribution
      - when n_trials = 1 and n_features > 2, Multinomial is the
        Categorical distribution
Multinomial HMM has the emission probabilities for each component parameterized
by the Multinomial distribution.
    - More details: https://en.wikipedia.org/wiki/Multinomial_distribution
"""
import jax.numpy as jnp
import jax.random as jr
from ssm_jax.hmm.learning import hmm_fit_em
from ssm_jax.hmm.models.multinomial_hmm import MultinomialHMM


def main():
    # For this example, we will model the stages of a conversation,
    # where each sentence is "generated" with an underlying topic, "cat" or "dog"
    states = ["cat", "dog"]

    # we are more likely to talk about cats first
    initial_probabilities = jnp.array([0.6, 0.4])

    # For each topic, the probability of saying certain words can be modeled by
    # a distribution over vocabulary associated with the categories

    vocabulary = ["tail", "fetch", "mouse", "food"]
    # if the topic is "cat", we are more likely to talk about "mouse"
    # if the topic is "dog", we are more likely to talk about "fetch"
    emission_probs = jnp.array([[0.25, 0.1, 0.4, 0.25], [0.2, 0.5, 0.1, 0.2]])

    # Also assume it's more likely to stay in a state than transition to the other
    transition_matrix = jnp.array([[0.8, 0.2], [0.2, 0.8]])

    # Pretend that every sentence we speak only has a total of 5 words,
    # i.e. we independently utter a word from the vocabulary 5 times per sentence
    # we observe the following bag of words (BoW) for 8 sentences:
    observations = [["tail", "mouse", "mouse", "food", "mouse"], ["food", "mouse", "mouse", "food", "mouse"],
                    ["tail", "mouse", "mouse", "tail", "mouse"], ["food", "mouse", "food", "food", "tail"],
                    ["tail", "fetch", "mouse", "food", "tail"], ["tail", "fetch", "fetch", "food", "fetch"],
                    ["fetch", "fetch", "fetch", "food", "tail"], ["food", "mouse", "food", "food", "tail"],
                    ["tail", "mouse", "mouse", "tail", "mouse"], ["fetch", "fetch", "fetch", "fetch", "fetch"]]

    # Convert "sentences" to numbers:
    vocab2id = dict(zip(vocabulary, range(len(vocabulary))))

    def sentence2ids(sentence):
        ans = []
        for obs in sentence:
            id = vocab2id[obs]
            ans.append(id)
        return ans

    X = []
    for sentence in observations:
        row = sentence2ids(sentence)
        X.append(row)

    data = jnp.array(X, dtype=int)

    # pretend this is repeated, so we have more data to learn from:
    sequences = jnp.tile(data, (5, 1))
    # Set up model
    model = MultinomialHMM(initial_probabilities, transition_matrix, emission_probs)

    hmm, log_probs = hmm_fit_em(model, sequences, num_iters=50)

    print("Learned emission probs:")
    print(hmm.emission_probs)

    print("Learned transition matrix:")
    print(hmm.transition_matrix)


if __name__ == "__main__":
    main()
