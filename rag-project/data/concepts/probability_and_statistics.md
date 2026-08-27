# Probability & Statistics — GATE DA reference notes

## Random variables and expectation
For a discrete random variable X with pmf p(x), the expectation is
E[X] = sum over x of x*p(x). Expectation is linear even when variables are
dependent: E[aX + bY] = a*E[X] + b*E[Y]. This linearity is the single most
useful shortcut in GATE probability questions, because it lets you decompose
a complicated count into a sum of indicator variables.

Variance is Var(X) = E[X^2] - (E[X])^2. Unlike expectation, variance is only
additive for *independent* variables: Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y),
and the covariance term vanishes only under independence.

## Standard distributions and their moments
- Bernoulli(p): mean p, variance p(1-p).
- Binomial(n, p): mean np, variance np(1-p). Models counts of successes in n
  independent trials.
- Poisson(lambda): mean lambda, variance lambda. The mean and variance being
  equal is a defining property and a frequent GATE test item.
- Uniform(a, b): mean (a+b)/2, variance (b-a)^2 / 12.
- Normal(mu, sigma^2): mean mu, variance sigma^2. The *standard* normal has
  mean 0 and variance 1; any normal is standardized by z = (x - mu)/sigma.
- Exponential(lambda): mean 1/lambda, variance 1/lambda^2. It is memoryless:
  P(X > s+t | X > s) = P(X > t).

## Conditional probability and Bayes' theorem
P(A|B) = P(A and B) / P(B), for P(B) > 0. Bayes' theorem rearranges this into
P(A|B) = P(B|A) * P(A) / P(B), where P(B) is usually expanded by the law of
total probability: P(B) = sum over i of P(B|A_i) * P(A_i) over a partition
{A_i}. Most "medical test" and "which urn did the ball come from" questions
are direct applications.

Independence means P(A and B) = P(A)*P(B). Independence is a stronger
condition than being mutually exclusive; in fact two events with nonzero
probability that are mutually exclusive are necessarily *dependent*, because
knowing one occurred forces the other to have probability zero.

## Counting for probability questions
For a family of n children with each birth independently a boy or a girl with
probability 1/2, the number of girls is Binomial(n, 1/2). The probability of
exactly k girls is C(n, k) / 2^n. For n = 3 and k = 2 this gives 3/8.

## Covariance and correlation
Cov(X, Y) = E[XY] - E[X]E[Y]. Correlation normalizes this to
rho = Cov(X,Y) / (sigma_X * sigma_Y), which always lies in [-1, 1].
Zero correlation does not imply independence — it only rules out a *linear*
relationship. Independence does imply zero correlation.

## Estimation
The sample mean is an unbiased estimator of the population mean. The sample
variance uses the (n-1) denominator (Bessel's correction) to remain unbiased;
dividing by n produces a biased estimator that systematically underestimates
the true variance.
