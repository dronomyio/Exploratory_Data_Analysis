# Solutions to Exercise 4.11, Questions 6 and 7

## Question 6: Standard Normal Cumulative Distribution Function

### Part (a): Finding Φ^(-1)(0.975)

Given that Φ^(-1)(0.025) = -1.96, we need to find Φ^(-1)(0.975).

**Solution:**
Φ^(-1)(0.975) = 1.96

**Explanation:**
The standard normal distribution is symmetric around 0. This means that:
- Φ(x) + Φ(-x) = 1 for any x
- Therefore, Φ^(-1)(1-p) = -Φ^(-1)(p) for any probability p

Since Φ^(-1)(0.025) = -1.96, we have:
Φ^(-1)(0.975) = Φ^(-1)(1-0.025) = -Φ^(-1)(0.025) = -(-1.96) = 1.96

This makes intuitive sense because the 0.025 and 0.975 quantiles are equidistant from the mean in a standard normal distribution, just in opposite directions.

### Part (b): Finding the 0.975-quantile of N(-1, 2)

We need to find the 0.975-quantile of a normal distribution with mean -1 and variance 2.

**Solution:**
The 0.975-quantile of N(-1, 2) = 1.77

**Explanation:**
For a normal distribution with mean μ and standard deviation σ:
- If Z ~ N(0,1) and X ~ N(μ,σ²), then X = μ + σZ
- So the p-quantile of X is: μ + σ * (p-quantile of Z)

In this case:
- μ = -1
- σ = √2 ≈ 1.414
- The 0.975-quantile of Z is 1.96 (from part a)

Therefore:
0.975-quantile of N(-1, 2) = -1 + 1.414 * 1.96 ≈ -1 + 2.77 = 1.77

## Question 7: Uniform Distribution and Sample Quantile with Minimum Variance

**Problem:**
For a uniform distribution on (0,1) with density function f(x) = 1 for x ∈ (0,1) and 0 otherwise, and distribution function F(x) = 0 if x ≤ 0, x if x ∈ (0,1), and 1 if x ≥ 1, determine which sample quantile q will have the smallest variance.

**Solution:**
The sample median (0.5-quantile) has the smallest variance among all sample quantiles for a uniform distribution on (0,1).

**Explanation:**
According to Result 4.1, the asymptotic variance of the sample p-quantile is:
σ²(p) = p(1-p)/[nf(F^(-1)(p))²]

For the uniform distribution on (0,1):
- F^(-1)(p) = p for p ∈ (0,1)
- f(F^(-1)(p)) = f(p) = 1 for p ∈ (0,1)

Therefore, the variance is proportional to p(1-p):
σ²(p) ∝ p(1-p)

To find the quantile with the smallest variance, we need to find the value of p that minimizes p(1-p).

Taking the derivative of p(1-p) with respect to p:
d/dp[p(1-p)] = 1-p - p = 1-2p

Setting this equal to zero:
1-2p = 0
p = 0.5

The second derivative is -2, which is negative, confirming this is a maximum.

Since p(1-p) is maximized at p = 0.5, and the variance is inversely proportional to p(1-p), the variance is minimized at p = 0.5.

Therefore, the sample median (0.5-quantile) has the smallest variance among all sample quantiles for a uniform distribution on (0,1).

This result makes intuitive sense because the uniform distribution has constant density throughout its support, and the median is the point that divides the distribution into equal halves, providing the most stable estimate.
