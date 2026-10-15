# The Math

Every claim the project relies on, proven. The first four build the main argument: why a covariance matrix estimated straight from history fails, and why a factor model fixes it. The rest back up the attribution and hedging.

Notation: $N$ stocks, $T$ days, $K$ factors. Vectors are columns. $w$ or $x$ is a portfolio (one weight per stock).

---

## 1. Definitions

A symmetric matrix $\Sigma$ is

- **positive semidefinite (PSD)** if $x^\top \Sigma x \ge 0$ for every vector $x$
- **positive definite (PD)** if $x^\top \Sigma x > 0$ for every vector $x \neq 0$

**Why a portfolio manager cares:** if $\Sigma$ is a covariance matrix of returns $r$, then $x^\top \Sigma x$ is the variance of portfolio $x$:

$$\text{Var}(x^\top r) = x^\top \Sigma x$$

So PSD means no portfolio has negative variance. PD means **no portfolio has zero variance either**: every combination of stocks carries some risk. If a matrix isn't PD, an optimizer will happily find a "riskless" portfolio that doesn't exist.

---

## 2. Every covariance matrix is PSD

**Claim.** Any covariance matrix is PSD.

**Proof.** For any $x$, $x^\top \Sigma x = \text{Var}(x^\top r)$, and a variance is never negative. ∎

The same holds for the **sample** covariance. With demeaned returns stacked in a $T \times N$ matrix $X$:

$$S = \frac{1}{T-1}X^\top X \qquad\Rightarrow\qquad x^\top S x = \frac{1}{T-1}(Xx)^\top(Xx) = \frac{\lVert Xx \rVert^2}{T-1} \ge 0 \qquad ∎$$

PSD comes for free. **PD does not**, which is the whole problem (Section 4).

---

## 3. PD ⟺ every eigenvalue is positive

**Claim.** A symmetric matrix is PD if and only if all its eigenvalues are $> 0$.

**Proof.** By the spectral theorem, a symmetric matrix can be written $\Sigma = Q \Lambda Q^\top$, where $Q$ is orthogonal ($Q^\top Q = I$, columns $q_k$ are the eigenvectors) and $\Lambda$ is diagonal with the eigenvalues $\lambda_1, \dots, \lambda_N$. Substitute $y = Q^\top x$:

$$x^\top \Sigma x = x^\top Q \Lambda Q^\top x = y^\top \Lambda y = \sum_{k=1}^{N} \lambda_k y_k^2$$

- (⇐) If every $\lambda_k > 0$ and $x \neq 0$, then $y \neq 0$ (because $Q$ is invertible), so at least one $y_k^2 > 0$ and the sum is $> 0$.
- (⇒) If $\Sigma$ is PD, plug in an eigenvector $x = q_k$: $q_k^\top \Sigma q_k = \lambda_k \lVert q_k \rVert^2 = \lambda_k$, which must be $> 0$. ∎

**Consequences used in the code:**

- $\det \Sigma = \prod_k \lambda_k > 0$, so $\Sigma^{-1}$ exists (needed for optimization).
- The Cholesky factorization $\Sigma = LL^\top$ exists exactly when $\Sigma$ is PD. That's how `is_positive_definite()` tests it: if Cholesky runs, the matrix is PD.
- Combined with Section 2: a PSD matrix has every $\lambda_k \ge 0$ (same algebra, with $\ge$).

---

## 4. The data problem: the sample covariance is singular when $N \ge T$

**Claim.** If there are at least as many stocks as days ($N \ge T$), the sample covariance $S$ is **not** PD. It has at least $N - T + 1$ eigenvalues equal to zero.

**Proof.**

1. **Demeaning costs one dimension.** Each column of $X$ has its mean subtracted, so each column sums to zero: $\mathbf{1}^\top X = 0$. That's a linear relationship between the $T$ rows, so $\text{rank}(X) \le T - 1$.

2. **$X^\top X$ has the same rank as $X$.** If $X^\top X x = 0$, then $x^\top X^\top X x = \lVert Xx \rVert^2 = 0$, so $Xx = 0$. The converse is obvious. Same null space, so same rank: $\text{rank}(S) = \text{rank}(X) \le T - 1$.

3. **$S$ is $N \times N$ with rank at most $T - 1 < N$.** By rank-nullity, its null space has dimension at least $N - (T-1)$, so there's a portfolio $x \neq 0$ with $Sx = 0$. Then $x^\top S x = 0$: **a portfolio with zero estimated risk.** Every null-space direction is an eigenvector with eigenvalue 0. ∎

**What this means:** 6,000 stocks need more than 6,000 trading days (about 24 years) before the sample covariance is even invertible. Most stocks don't have that history, and their relationships don't stay stable that long. And even with $T > N$, an $N(N+1)/2$-number matrix built from limited data is mostly noise (Section 9).

**See it in Excel:** take 30 stocks and 20 days of returns, build the covariance matrix with `COVARIANCE.S`, then try `MINVERSE`. It errors out; `MDETERM` comes back as 0. In code: `test_direct_covariance_breaks_with_more_stocks_than_days`.

---

## 5. The fix: a factor-model covariance is always PD

**Setup.** Each stock's return is explained by $K$ shared factors plus noise of its own:

$$r_i = \alpha_i + \beta_i^\top f + \varepsilon_i, \qquad \text{Cov}(\varepsilon_i, \varepsilon_j) = 0 \text{ for } i \neq j$$

Taking covariances (the $\alpha_i$ are constants and drop out):

$$\text{Cov}(r_i, r_j) = \beta_i^\top F \beta_j \;\; (i \neq j), \qquad \text{Var}(r_i) = \beta_i^\top F \beta_i + d_i$$

In matrix form, with $B$ ($N \times K$) stacking the betas, $F$ the factor covariance, and $D = \text{diag}(d_1, \dots, d_N)$ the specific variances:

$$\boxed{\Sigma = BFB^\top + D}$$

Two different stocks are correlated **only through the factors they share**. That's the assumption that cuts $N(N+1)/2$ unknowns down to $NK + K(K+1)/2 + N$.

**Claim.** If $F$ is PSD and every $d_i > 0$, then $\Sigma$ is PD, **no matter how $N$ compares to $T$**.

**Proof.** Take any $x \neq 0$ and let $b = B^\top x$:

$$x^\top \Sigma x = \underbrace{b^\top F b}_{\ge\, 0 \text{ since } F \text{ is PSD}} + \underbrace{\sum_{i=1}^{N} d_i x_i^2}_{>\, 0 \text{ since } x \neq 0 \text{ and every } d_i > 0} > 0 \qquad ∎$$

$F$ is PSD automatically: it's a sample covariance of the factors (Section 2). It's a $K \times K$ matrix, so it only needs $T > K$ days, which is trivial for $K = 6$.

**Each model meets the conditions:**

- **CAPM and Fama-French:** $F$ is the factors' sample covariance (PSD). Each $d_i$ is a regression's residual variance, which is $> 0$ unless the factors explain a stock perfectly, which never happens with real returns.
- **Statistical (PCA):** $F = \text{diag}(\lambda_1, \dots, \lambda_K)$ with $\lambda_k \ge 0$ (PSD). For $D$, write the full sample covariance as $S = \sum_k \lambda_k q_k q_k^\top$, so its diagonal is $S_{ii} = \sum_k \lambda_k q_{ik}^2$. Keeping the top $K$ factors leaves

$$d_i = S_{ii} - \sum_{k \le K} \lambda_k q_{ik}^2 = \sum_{k > K} \lambda_k q_{ik}^2 \ge 0$$

It's $\ge 0$ rather than strictly $> 0$, so the code floors each $d_i$ at a tiny positive number to guarantee PD.

---

## 6. Return attribution adds up exactly

Regress the portfolio's excess return on the factors: $r_{p,t} = \hat\alpha + \hat b^\top f_t + e_t$.

**Claim.** $\bar r_p = \hat\alpha + \sum_k \hat b_k \bar f_k$, so the factor contributions plus alpha equal the average return exactly.

**Proof.** OLS chooses $\hat\alpha$ to minimize $\sum_t e_t^2$. Setting the derivative with respect to $\hat\alpha$ to zero gives $-2\sum_t e_t = 0$, so the residuals average to exactly zero. Average both sides of the regression: $\bar r_p = \hat\alpha + \hat b^\top \bar f + 0$. ∎

---

## 7. Risk attribution adds up exactly

**Claim.** $\text{Var}(r_p) = \hat b^\top F \hat b + \text{Var}(e)$, and the factor part splits into pieces $c_k = \hat b_k (F\hat b)_k$ that sum to it.

**Proof.** OLS also sets the derivative with respect to each $\hat b_k$ to zero, giving $\sum_t e_t f_{k,t} = 0$. Combined with $\bar e = 0$, the residuals are uncorrelated with every factor, so the cross term vanishes:

$$\text{Var}(\hat b^\top f + e) = \hat b^\top F \hat b + \text{Var}(e) + 2\,\underbrace{\text{Cov}(\hat b^\top f,\, e)}_{0}$$

And $\sum_k \hat b_k (F\hat b)_k = \hat b^\top F \hat b$ by the definition of the dot product. ∎

---

## 8. Position risk contributions add to 100%

**Claim.** With $RC_i = \dfrac{w_i (\Sigma w)_i}{w^\top \Sigma w}$, the contributions sum to 1.

**Proof.** $\sum_i w_i (\Sigma w)_i = w^\top \Sigma w$. Divide both sides by $w^\top \Sigma w$. ∎

This is Euler's theorem in disguise. Portfolio volatility $\sigma(w) = \sqrt{w^\top \Sigma w}$ scales linearly ($\sigma(cw) = c\,\sigma(w)$), and Euler's theorem says such functions satisfy $\sigma = \sum_i w_i \frac{\partial \sigma}{\partial w_i}$, with $\frac{\partial \sigma}{\partial w} = \frac{\Sigma w}{\sigma}$. Each term is how much risk that position adds at the margin, times its size.

---

## 9. How many factors are real? (Marchenko-Pastur)

If returns were pure independent noise, the eigenvalues of an $N \times N$ sample **correlation** matrix built from $T$ days would, for large $N$ and $T$, all fall inside

$$\lambda_\pm = \left(1 \pm \sqrt{N/T}\right)^2$$

This is a result from random matrix theory (Marchenko and Pastur, 1967). It's quoted here, not proven. Eigenvalues **above** $\lambda_+$ can't be explained by noise, so they're real factors. The statistical model keeps exactly those.

---

## 10. Effective number of bets

For a correlation matrix, the eigenvalues sum to $N$ (the trace: every diagonal entry is 1). Define

$$N_{\text{eff}} = \frac{\left(\sum_k \lambda_k\right)^2}{\sum_k \lambda_k^2}$$

**Claim.** $1 \le N_{\text{eff}} \le N$.

**Proof.** Upper bound: by Cauchy-Schwarz, $\left(\sum_k 1 \cdot \lambda_k\right)^2 \le N \sum_k \lambda_k^2$. Lower bound: all $\lambda_k \ge 0$, so $\left(\sum_k \lambda_k\right)^2 = \sum_k \lambda_k^2 + \sum_{j \neq k} \lambda_j \lambda_k \ge \sum_k \lambda_k^2$. ∎

Independent stocks (every $\lambda_k = 1$) give $N_{\text{eff}} = N$. Everything moving as one (one eigenvalue $= N$, the rest 0) gives $N_{\text{eff}} = 1$.

---

## 11. The best hedge ratio

Short $h$ dollars of an ETF per dollar of portfolio. The hedged variance is

$$V(h) = \text{Var}(r_p - h r_e) = \sigma_p^2 - 2h\,\sigma_{pe} + h^2 \sigma_e^2$$

**Claim.** $V$ is minimized at $h^* = \sigma_{pe}/\sigma_e^2$, leaving $V(h^*) = \sigma_p^2(1 - \rho^2)$.

**Proof.** $V'(h) = -2\sigma_{pe} + 2h\sigma_e^2 = 0$ gives $h^* = \sigma_{pe}/\sigma_e^2$. $V''(h) = 2\sigma_e^2 > 0$, so it's a minimum. Plug it back in:

$$V(h^*) = \sigma_p^2 - 2\frac{\sigma_{pe}^2}{\sigma_e^2} + \frac{\sigma_{pe}^2}{\sigma_e^2} = \sigma_p^2 - \frac{\sigma_{pe}^2}{\sigma_e^2} = \sigma_p^2\left(1 - \frac{\sigma_{pe}^2}{\sigma_p^2\sigma_e^2}\right) = \sigma_p^2(1 - \rho^2) \qquad ∎$$

So the ETF that removes the most risk is the one with the highest correlation to the book, and $\rho^2$ is the share of variance it removes. $h^*$ is also the slope from regressing the book on the ETF, the same formula as a beta.
