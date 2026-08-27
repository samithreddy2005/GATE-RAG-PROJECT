# Calculus & Optimization — GATE DA reference notes

## Continuity and differentiability
A function is continuous at a point if the left limit, right limit and the
function value all coincide. Differentiability implies continuity, but not the
reverse: |x| is continuous everywhere and not differentiable at 0. This
one-way implication is a standard true/false item.

## Extrema
At an interior local extremum of a differentiable function the derivative is
zero — a *necessary* condition, not a sufficient one, as x^3 at 0 shows.
The second-derivative test classifies a stationary point: f''>0 gives a local
minimum, f''<0 a local maximum, f''=0 is inconclusive.

For a function of several variables, stationary points satisfy
gradient = 0, and the Hessian classifies them: positive definite gives a local
minimum, negative definite a local maximum, indefinite a saddle point.

On a closed bounded interval, the global extrema of a continuous function are
attained either at a stationary point or at an endpoint — endpoints are the
most commonly forgotten half.

## Convexity
A function is convex if its Hessian is positive semi-definite everywhere,
equivalently if the chord between any two points lies on or above the graph.
The decisive property for optimization: for a convex function on a convex set,
**every local minimum is a global minimum**, and the set of minimizers is
convex. Without convexity, gradient descent guarantees only a local optimum.

## Gradient descent
The update is x <- x - eta * grad f(x). The gradient points in the direction
of steepest *increase*, hence the minus sign. The learning rate eta controls
convergence: too small converges slowly, too large can oscillate or diverge.
For a convex differentiable function with a suitably small fixed step size,
gradient descent converges to the global minimum.

Stochastic gradient descent estimates the gradient from a minibatch. It is
noisier per step but far cheaper, and the noise can help escape saddle points
in non-convex problems such as neural network training.

## Series and limits
A geometric series sum of a*r^n from n=0 converges to a/(1-r) iff |r| < 1.
The harmonic series sum of 1/n diverges, while sum of 1/n^p converges iff
p > 1. Splitting a mixed series into recognizable geometric parts is the usual
route through GATE infinite-series questions: for example
2 + 1/2 + 1/3 + 1/4 + 1/8 + 1/9 + 1/16 + 1/27 + ... separates into the powers
of 1/2 and the powers of 1/3, each summable independently as a geometric
series.

L'Hopital's rule applies only to indeterminate forms 0/0 and infinity/infinity;
applying it elsewhere produces wrong answers.

## Integration
The fundamental theorem links differentiation and integration:
d/dx of the integral from a to x of f(t) dt equals f(x) for continuous f.
Definite integrals of odd functions over a symmetric interval [-a, a] vanish;
for even functions they equal twice the integral over [0, a] — a shortcut that
turns several exam integrals into one line.
