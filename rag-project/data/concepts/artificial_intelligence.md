# Artificial Intelligence — GATE DA reference notes

## Uninformed search
- BFS expands the shallowest node. Complete, and optimal when every edge has
  the same cost. Time and space are both O(b^d).
- DFS expands the deepest node. Space is only O(b*m), but it is neither
  complete (on infinite branches) nor optimal.
- Uniform-cost search expands the lowest path cost g(n) and is optimal for
  non-negative edge costs.
- Iterative deepening (IDDFS) runs depth-limited DFS with increasing limits.
  It combines the space efficiency of DFS with the completeness and optimality
  of BFS, and the repeated work is only a constant factor because the number
  of nodes at the deepest level dominates the total.

## Informed search and heuristics
A* expands the node minimizing f(n) = g(n) + h(n), the cost so far plus the
estimated cost to the goal.

A heuristic h is **admissible** if it never overestimates the true remaining
cost: h(n) <= h*(n) for all n. A* with an admissible heuristic is optimal for
tree search.

A heuristic is **consistent** (monotone) if h(n) <= c(n, n') + h(n') for every
successor n'. Consistency implies admissibility, and A* with a consistent
heuristic is optimal for *graph* search, where nodes may be re-reached by a
cheaper path. The converse does not hold: an admissible heuristic need not be
consistent.

If h1 and h2 are both admissible, max(h1, h2) is admissible and dominates
both, meaning it expands no more nodes than either. h(n) = 0 reduces A* to
uniform-cost search.

## Adversarial search
Minimax computes the optimal move assuming an optimal opponent, propagating
maximizing and minimizing values alternately up the game tree.

Alpha-beta pruning returns exactly the same value as minimax — it changes
efficiency, never the result. Alpha is the best value the maximizer can
guarantee so far; beta the best the minimizer can guarantee. A branch is
pruned when alpha >= beta. With perfect move ordering the effective branching
factor drops from b to roughly sqrt(b), so the searchable depth roughly
doubles for the same work; with random ordering the gain is smaller.

## Propositional and first-order logic
A formula is a **tautology** if it is true under every interpretation,
**satisfiable** if true under at least one, and **unsatisfiable** if true
under none. A is valid iff not-A is unsatisfiable — the identity that makes
proof by refutation work.

Key equivalences: implication A -> B is equivalent to (not A) or B; its
contrapositive (not B) -> (not A) is equivalent to it, while its converse
B -> A is not. De Morgan: not(A and B) = (not A) or (not B).

In first-order logic, quantifier order matters: "for all x, exists y, P(x,y)"
is weaker than "exists y, for all x, P(x,y)". Negation flips quantifiers:
not(for all x, P(x)) is equivalent to (exists x, not P(x)).

## Bayesian networks
A directed acyclic graph where each node is conditionally independent of its
non-descendants given its parents. The joint distribution factorizes as
P(X1..Xn) = product of P(Xi | parents(Xi)), which is what makes inference
tractable relative to a full joint table. A node with k binary parents needs
2^k rows in its conditional probability table.
