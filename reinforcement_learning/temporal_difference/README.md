# Temporal Difference

Reinforcement learning: Monte Carlo, TD(λ), and SARSA(λ) for prediction and control.

## Tasks

| #   | File                 | Description                              |
| --- | -------------------- | ---------------------------------------- |
| 0   | `0-monte_carlo.py`    | Monte Carlo policy evaluation            |
| 1   | `1-td_lambtha.py`     | TD(λ) with eligibility traces            |
| 2   | `2-sarsa_lambtha.py`  | SARSA(λ) with eligibility traces         |

## Requirements

- Python 3.5
- NumPy 1.15, gym 0.7

## Usage

Run main files from the project description or import:

```python
monte_carlo = __import__('0-monte_carlo').monte_carlo
td_lambtha = __import__('1-td_lambtha').td_lambtha
sarsa_lambtha = __import__('2-sarsa_lambtha').sarsa_lambtha
```
