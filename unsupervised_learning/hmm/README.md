# Hidden Markov Models

Unsupervised learning: Markov chains, regular/absorbing properties, and Hidden Markov Models (forward, backward, Viterbi, Baum-Welch).

## Tasks

| #   | File              | Description                                              |
| --- | ----------------- | -------------------------------------------------------- |
| 0   | `0-markov_chain.py` | Probability of being in each state after t iterations  |
| 1   | `1-regular.py`    | Steady state probabilities of a regular Markov chain     |
| 2   | `2-absorbing.py`  | Determine if a Markov chain is absorbing                 |
| 3   | `3-forward.py`    | Forward algorithm: likelihood and forward path probs     |
| 4   | `4-viterbi.py`    | Viterbi algorithm: most likely sequence of hidden states |
| 5   | `5-backward.py`   | Backward algorithm: likelihood and backward path probs   |
| 6   | `6-baum_welch.py` | Baum-Welch algorithm: estimate Transition and Emission  |

## Requirements

- Python 3.5
- NumPy 1.15

## Usage

Run main files from the project description (e.g. `./3-main.py`) or import functions:

```python
forward = __import__('3-forward').forward
viterbi = __import__('4-viterbi').viterbi
backward = __import__('5-backward').backward
baum_welch = __import__('6-baum_welch').baum_welch
```
