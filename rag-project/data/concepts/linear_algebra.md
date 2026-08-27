# Linear Algebra — GATE DA reference notes

## Eigenvalues and eigenvectors
For a square matrix M, a scalar lambda is an eigenvalue if M*v = lambda*v for
some nonzero vector v. Eigenvalues are the roots of the characteristic
polynomial det(M - lambda*I) = 0.

Two identities make most GATE eigenvalue questions solvable without fully
factoring the polynomial:
- The sum of the eigenvalues equals the trace of M (the sum of its diagonal).
- The product of the eigenvalues equals det(M).

For a 2x2 matrix the characteristic polynomial is
lambda^2 - trace(M)*lambda + det(M) = 0. Its discriminant is
trace^2 - 4*det. If the discriminant is negative, the eigenvalues are a
complex conjugate pair. Because a real matrix has a real characteristic
polynomial, complex eigenvalues of a real matrix *always* occur in conjugate
pairs — one can never be complex while the other is real.

Worked example: M = [[2, -1], [3, 1]]. trace = 3, det = 2*1 - (-1)*3 = 5.
Discriminant = 9 - 20 = -11 < 0, so the eigenvalues are complex conjugates.

## Symmetric matrices
A real symmetric matrix always has real eigenvalues and an orthogonal set of
eigenvectors (the spectral theorem). A symmetric matrix is positive definite
iff all its eigenvalues are strictly positive, equivalently iff x^T M x > 0
for every nonzero x. Positive semi-definite relaxes this to >= 0. Covariance
matrices are always symmetric positive semi-definite, which is why PCA is
always well defined.

## Rank, null space and solvability
rank(M) is the number of linearly independent rows, equal to the number of
linearly independent columns. The rank-nullity theorem states
rank(M) + nullity(M) = n for an m x n matrix, where nullity is the dimension
of the null space {x : Mx = 0}.

For M*x = b: the system is consistent iff rank(M) = rank([M | b]). If
consistent, the solution is unique iff rank(M) equals the number of columns,
and otherwise there are infinitely many solutions.

A square matrix is invertible iff det != 0 iff rank is full iff zero is not an
eigenvalue iff the null space contains only the zero vector. These are all the
same statement.

## Singular value decomposition
Any real m x n matrix factors as M = U * Sigma * V^T, where U and V are
orthogonal and Sigma is diagonal with non-negative singular values. The
singular values of M are the square roots of the eigenvalues of M^T*M. SVD
underpins Truncated SVD / LSA, which is exactly the embedding technique used
by this project's local embedder.

## Principal component analysis
PCA projects data onto the eigenvectors of the covariance matrix, ordered by
descending eigenvalue. The eigenvalue associated with a principal component is
the variance explained along that direction, so the fraction of total variance
retained by the top k components is (sum of top k eigenvalues) / (sum of all
eigenvalues). Data must be mean-centered first, otherwise the first component
merely points at the mean.
