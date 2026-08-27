# DBMS & Data Warehousing — GATE DA reference notes

## Functional dependencies and keys
X -> Y means that any two tuples agreeing on X must agree on Y. Armstrong's
axioms are reflexivity, augmentation and transitivity; from these follow union,
decomposition and pseudo-transitivity.

The **attribute closure** X+ is the set of all attributes functionally
determined by X. X is a superkey iff X+ is the full relation schema, and a
candidate key iff it is a superkey with no proper subset that is also a
superkey. Computing closures is the mechanical way to answer almost every
GATE key-identification question.

## Normal forms
- **1NF**: all attribute values are atomic.
- **2NF**: 1NF plus no partial dependency of a non-prime attribute on part of
  a candidate key. Only relevant when a candidate key is composite.
- **3NF**: 2NF plus, for every dependency X -> A, either X is a superkey or A
  is prime.
- **BCNF**: for every non-trivial X -> A, X must be a superkey.

Every BCNF relation is in 3NF, but not conversely. A lossless-join
decomposition into BCNF always exists; a *dependency-preserving* BCNF
decomposition does not always exist, which is exactly why 3NF still gets used.

A binary decomposition of R into R1 and R2 is lossless iff
(R1 intersect R2) -> R1 or (R1 intersect R2) -> R2.

## Relational algebra and SQL
Core operators: selection (sigma), projection (pi), Cartesian product, union,
set difference, rename; joins are derived. Projection removes duplicates in
relational algebra, whereas SQL's `SELECT` keeps them unless `DISTINCT` is
given — a difference GATE tests repeatedly.

`WHERE` filters rows before grouping; `HAVING` filters groups after. Aggregate
functions ignore NULLs, except `COUNT(*)` which counts rows. Any comparison
with NULL yields UNKNOWN, so `WHERE x = NULL` never matches; `IS NULL` is
required.

An SQL `NOT IN` against a subquery containing a NULL returns no rows at all,
because the comparison evaluates to UNKNOWN rather than TRUE — a common
correctness bug and a favourite exam item.

## Transactions
ACID: Atomicity, Consistency, Isolation, Durability.

A schedule is **conflict serializable** if its precedence graph is acyclic.
Two-phase locking (growing then shrinking phase) guarantees conflict
serializability but can deadlock; strict 2PL, which holds all locks until
commit, additionally guarantees recoverability and avoids cascading aborts.

Isolation levels and the anomalies they permit: READ UNCOMMITTED allows dirty
reads; READ COMMITTED prevents dirty reads but allows non-repeatable reads;
REPEATABLE READ still allows phantoms; SERIALIZABLE prevents all three.

## Indexing
A **primary/clustered** index orders the actual data file, so there can be
only one per table; range queries on it are cheap. A **secondary** index is
unclustered and needs an extra lookup per match.

A dense index has an entry per record; a sparse index has one per block and
therefore requires a clustered file. B+ trees dominate because their height
determines the number of disk accesses, and high fan-out keeps height at 3-4
even for very large tables.

## Data warehousing
OLTP workloads are many small writes, highly normalized. OLAP workloads are
few large aggregating reads, deliberately denormalized into star schemas (one
fact table plus dimension tables) to avoid join cost. A snowflake schema
normalizes the dimensions, trading query speed for storage and consistency.
Typical OLAP operations are roll-up, drill-down, slice, dice and pivot.
