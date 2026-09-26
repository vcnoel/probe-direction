# Mathematical notes

This note separates algebraic facts from empirical hypotheses. Let `X ∈ R^{n×d}` be a token-by-hidden signal and let `W ∈ R^{n×n}` be a fixed symmetric nonnegative adjacency. Unless stated otherwise, isolated nodes are removed or handled as separate zero-degree components.

## 1. Fixed-combinatorial DC invariance

Define `D = diag(W1)` and the combinatorial Laplacian `L = D - W`. Its Dirichlet energy is

`E_L(X) = tr(X^T L X)`.

For any hidden-space vector `a ∈ R^d`,

`E_L(X + 1a^T) = E_L(X)`.

**Proof.** Symmetry gives `L1 = 0` and `1^T L = 0`. Expanding,

`tr((X+1a^T)^T L (X+1a^T))`

`= tr(X^T L X) + 2 tr(a 1^T L X) + tr(a 1^T L 1 a^T)`.

The latter terms vanish. Therefore the energy is invariant. The qualification **fixed-combinatorial** matters: W and hence L must be held fixed. If changing the prompt also changes W, this identity does not equate energies across conditions.

## 2. Symmetric-normalized null mode is degree weighted

The default basis-producing operator is

`L_sym = I - D^{-1/2} W D^{-1/2}`.

Let `q = D^{1/2}1`. Then

`L_sym q = D^{1/2}1 - D^{-1/2}W1 = D^{1/2}1 - D^{-1/2}D1 = 0`.

Thus the null mode is proportional to `D^{1/2}1`, not generally to `1`. For a connected graph with positive degrees, the normalized null eigenvector is `u_0 = q / ||q||_2`. The null coefficient of X is `u_0^T X`, and the lifted null component is

`X_null = u_0 u_0^T X`.

Adding `q a^T` leaves `tr(X^T L_sym X)` invariant by the same expansion. Adding an ordinary token-constant `1a^T` is not guaranteed to do so unless degrees are constant. For disconnected graphs, each connected component contributes a degree-weighted null vector.

## 3. Lifted DC/AC Frobenius accounting

For ordinary token-mean decomposition, define

`μ = (1/n)1^T X`,

`X_DC = 1 μ`, and `X_AC = X - X_DC`.

Because `1^T X_AC = 0`, the two matrices are Frobenius orthogonal. Therefore

`||X||_F^2 = ||X_DC||_F^2 + ||X_AC||_F^2`

`= n ||μ||_2^2 + ||X_AC||_F^2`.

The factor n is essential: comparing `||μ||_2^2` directly with `||X_AC||_F^2` mixes an averaged vector with a lifted token-space matrix. A DC mass fraction is

`n||μ||_2^2 / ||X||_F^2`.

For the symmetric-normalized basis, replace ordinary DC with the orthogonal degree-weighted projection:

`X_null = q (q^T X)/(q^T q)`, and `X_perp = X - X_null`.

Then

`||X||_F^2 = ||X_null||_F^2 + ||X_perp||_F^2`

and

`||X_null||_F^2 = ||q||_2^2 ||(q^T X)/(q^Tq)||_2^2`.

Since `||q||_2^2 = Σ_i d_i`, the lift factor is total degree rather than token count. Ordinary DC/AC and normalized-null/perpendicular decompositions answer different questions and must be named separately.

## 4. Valid spectral accounting

Because `L_sym` is real symmetric, it has an orthonormal eigenbasis `U` and real nonnegative eigenvalues `Λ`. With `X_hat = U^T X`, Parseval and Dirichlet accounting hold:

`||X||_F^2 = Σ_k ||X_hat_k||_2^2`,

`tr(X^T L_sym X) = Σ_k λ_k ||X_hat_k||_2^2`.

The categorical random-walk operator `L_rw = I - D^{-1}W` is generally nonsymmetric. Its right eigenvectors are not generally Euclidean orthonormal. Therefore ordinary projections onto those vectors do not justify Parseval, orthogonal DC/AC mass, spectral entropy, high-frequency energy ratios, or Euclidean principal-angle claims. Similarity of `L_rw` to `L_sym` does not repair code that treats random-walk right eigenvectors as an orthonormal basis. Basis-dependent analysis must use `L_sym` or an explicitly derived weighted inner product.

## 5. Non-invariances and non-theorems

The following statements do **not** follow from the identities above:

- **Changing graph:** DC invariance for fixed L does not imply invariance when the condition changes W, D, node set, head aggregation, or token span.
- **Normalized ordinary DC:** `L_sym` does not generally annihilate `1`; ordinary mean shifts need not be invisible.
- **All spectral metrics:** invariance of Dirichlet energy does not automatically make every normalized ratio, entropy, thresholded statistic, eigengap, or graph-only metric invariant.
- **Attention mechanism:** a change in W does not prove a change in transmitted information; values and output projections remain uncontrolled.
- **Recognition:** decodability of E from X does not prove R, belief, intent, or awareness.
- **Behavior or causality:** an association between a representation and P does not prove mediation. Conversely, a null linear intervention does not prove the representation is irrelevant.
- **Frequency story:** fixed-operator algebra does not imply that nuisance shifts are low frequency or that evaluation-related shifts are high frequency. That ordering is an empirical conjecture.
- **Scaling law:** monotonic values across a few checkpoints do not establish a law, separate scale from family, or identify a mechanism.

## 6. Three comparison modes

Every spectral comparison must declare one of these modes.

### Graph-only

Compare operators or graph summaries across conditions without projecting a hidden-state signal. Examples include degree distributions, Laplacian eigenvalues, or connectivity. This mode supports graph-structure statements only.

### Fixed-reference-signal

Choose one preregistered signal `X_ref` and evaluate it against each condition’s graph, or choose one preregistered graph and project each condition’s signal onto that common basis. The frozen side must come from training data or a symmetric construction that does not expose confirmation labels. This mode isolates graph change from signal change, subject to the chosen reference.

### Joint graph-and-signal

Evaluate each condition’s signal on its own condition-specific graph. This measures the coupled pipeline and is often the operational quantity, but differences cannot be attributed to graph or signal alone. Value/output controls are still required for an attention-routing interpretation.

Report all available modes together when making a mechanistic attribution. Never compare coefficients from unrelated eigenbases without an alignment rule, and use orientation-invariant statistics for eigenvectors or learned directions.
