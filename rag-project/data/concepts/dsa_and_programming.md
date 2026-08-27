# Data Structures, Algorithms & Programming — GATE DA reference notes

## Asymptotic notation
O(f) is an upper bound, Omega(f) a lower bound, Theta(f) a tight bound.
"Worst case" and "upper bound" are independent axes: it is perfectly correct
to state a lower bound on the worst case. Common growth order:
1 < log n < sqrt(n) < n < n log n < n^2 < n^3 < 2^n < n!.

The Master theorem for T(n) = a*T(n/b) + f(n) compares f(n) with
n^(log_b a): if f is polynomially smaller the recursion cost dominates and
T = Theta(n^(log_b a)); if they match, T = Theta(n^(log_b a) * log n); if f is
polynomially larger and the regularity condition holds, T = Theta(f(n)).

## Sorting
- Quicksort: average O(n log n), worst case O(n^2) when the pivot is
  consistently extreme (already-sorted input with a first-element pivot).
  In-place, not stable.
- Mergesort: O(n log n) in every case, stable, needs O(n) extra space.
- Heapsort: O(n log n) worst case, in place, not stable.
- Counting/radix sort beat the O(n log n) comparison lower bound only because
  they are not comparison-based; they assume bounded integer keys.

Any comparison sort needs Omega(n log n) comparisons, from the decision-tree
argument: n! leaves require depth at least log2(n!) = Theta(n log n).

## Core structures
- **Stack** is LIFO, **queue** is FIFO. Both give O(1) push/pop.
- **Binary heap**: complete binary tree with the heap property. Insert and
  extract-min are O(log n); find-min is O(1); *building* a heap from n
  elements is O(n), not O(n log n) — a standard GATE trap.
- **BST**: O(h) search/insert/delete where h is height. Balanced (AVL,
  red-black) guarantees h = O(log n); an unbalanced BST degrades to O(n).
  An in-order traversal of a BST yields sorted order.
- **Hash table**: O(1) average lookup, O(n) worst case when everything
  collides. Load factor drives performance; chaining and open addressing
  handle collisions differently.
- **B+ tree**: all data lives in leaves, which are linked, making range scans
  efficient. High fan-out keeps the height small, which is why it is the
  standard database index structure — height determines disk seeks.

## Graph algorithms
- BFS computes shortest paths in an **unweighted** graph; DFS classifies edges
  and finds cycles and topological order.
- Dijkstra needs non-negative weights; Bellman-Ford tolerates negative edges
  and detects negative cycles, in O(V*E).
- A topological order exists iff the graph is a DAG.
- Kruskal and Prim both produce a minimum spanning tree. When all edge weights
  are distinct, the MST is unique.

## Programming semantics that GATE tests
Python integer division `//` floors toward negative infinity, so `-7 // 2` is
-4, not -3, while `int(-7/2)` truncates toward zero and gives -3.

Mutable default arguments in Python are evaluated once at function definition
time, so `def f(x, acc=[])` shares the same list across calls — a classic
source of surprising output in trace questions.

Lists are passed by object reference: rebinding the name inside a function
does not affect the caller, but mutating the object does.
