# Probability Project

This project explores key concepts in probability and statistics by implementing classes for common probability distributions. These include Poisson, Exponential, Normal, and Binomial distributions. Each class is built from scratch using only built-in Python functionalities—no external libraries like `math` or `numpy` are used.

## 📚 Learning Objectives

By the end of this project, I was able to:

- Understand and explain what probability is
- Use basic probability notation
- Distinguish between independent and disjoint events
- Apply union and intersection rules
- Use the general addition and multiplication rules
- Understand probability distributions, PDFs, PMFs, and CDFs
- Calculate percentiles, mean, variance, and standard deviation
- Identify and work with Poisson, Exponential, Normal, and Binomial distributions

---

## 🛠️ Implemented Distributions

### 1. Poisson Distribution (`poisson.py`)

- Initialize using raw data or λ (expected number of events)
- Calculate PMF and CDF

### 2. Exponential Distribution (`exponential.py`)

- Initialize using raw data or λ (rate parameter)
- Calculate PDF and CDF

### 3. Normal Distribution (`normal.py`)

- Initialize using raw data or given mean and standard deviation
- Calculate z-score and corresponding x-value
- Compute PDF and CDF using error function approximation

### 4. Binomial Distribution (`binomial.py`)

- Initialize using raw data or given number of trials `n` and probability `p`
- Calculate PMF and CDF

---

## 🧪 Usage

Each distribution class can be tested using the provided `*-main.py` test scripts. All classes are initialized using either raw data or parameter values and include methods to compute PMF/PDF and CDF values.

Example (Poisson):

```bash
./0-main.py
```
