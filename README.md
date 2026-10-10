# Air Pollution Prediction with Local Gaussian Processes

Predicts PM2.5 concentration at unobserved city locations from 15,189 noisy
measurements, under an asymmetric cost that penalises underprediction inside
candidate residential zones 50× more than any other error.

## Approach

We observe $y_i = f(x_i) + \varepsilon_i$ at coordinates $x_i \in [0,1]^2$ and must
predict $f$ elsewhere. The loss is

$$
\ell_W(f, \hat{f}) = (f - \hat{f})^2 \cdot
\begin{cases}
50 & \hat{f} \le f \text{ and } x \text{ residential} \\
1 & \text{otherwise}
\end{cases}
$$

Exact GP inference costs $O(n^3)$ per marginal-likelihood evaluation, so at
$n = 15189$ fitting hyperparameters globally is infeasible. The model is therefore a
collection of local GPs, followed by a correction for the asymmetry.

**1. Partition.** $k$-means splits the map into $K = 85$ clusters $C_k$ with
centroids $m_k$. Cluster $k$ trains on its own members plus a halo of neighbours,

$$
I_k = \lbrace i : \|x_i - m_k\| \le (1 + c) r_k \rbrace
\qquad
r_k = \max_{j \in C_k} \|x_j - m_k\|
$$

with $c = 0.35$, so a point on a cluster edge still has data beyond it. Sets larger
than 1500 points are subsampled, keeping members over halo points.

**2. Fit.** Each cluster gets an independent GP with a Matérn kernel of smoothness
$\nu = \frac{5}{2}$, where $d = \|x - x'\|$:

$$
k(x, x') = \sigma^2 \left( 1 + \frac{d\sqrt{5}}{\ell} + \frac{5 d^2}{3 \ell^2} \right)
e^{-d\sqrt{5}/\ell} + \sigma_n^2 \delta_{x x'}
$$

The triple $(\sigma, \ell, \sigma_n)$ is fitted per cluster by maximising the log
marginal likelihood, letting each region adapt its own amplitude and noise level.
This $\nu$ is used because the field is rougher than a squared-exponential kernel
can represent, and smoother than $\nu = \frac{3}{2}$ assumes.

**3. Predict.** A test point is routed to its nearest centroid, yielding a posterior
mean $\mu(x)$ and standard deviation $s(x)$, with $s^2 = s_f^2 + \sigma_n^2$ for
latent uncertainty $s_f$.

**4. Shift for the asymmetry.** For $f(x) \mid \mathcal{D} \sim \mathcal{N}(\mu, s^2)$
and a prediction $\hat{f} = \mu + z s$, writing $t \sim \mathcal{N}(0,1)$:

$$
\frac{\mathbb{E}[\ell_W]}{s^2} = w A(z) + B(z)
\qquad
\begin{aligned}
A(z) &= \mathbb{E}\left[(t - z)^2 \mathbf{1}[t \ge z]\right] = (1 + z^2)(1 - \Phi(z)) - z \phi(z) \\
B(z) &= \mathbb{E}\left[(t - z)^2 \mathbf{1}[t < z]\right] = (1 + z^2) \Phi(z) + z \phi(z)
\end{aligned}
$$

Here $s^2$ factors out, so the optimal shift is a single constant for the whole map:
the root of $w A'(z) + B'(z) = 0$, that is

$$
w \left( z (1 - \Phi(z)) - \phi(z) \right) + z \Phi(z) + \phi(z) = 0
$$

which gives $z^{\ast} = 1.4852$ for $w = 50$. The loss is convex in $z$, so the root
is unique.

**5. Discount the noise.** $s$ is the predictive standard deviation of a *noisy*
observation, while the evaluation targets are noiseless. Measurement noise should
therefore not be insured against, but field structure below the sampling resolution
should, since it persists in the target. A fraction $\beta = 0.45$ of the fitted
noise is retained, calibrated empirically:

$$
\tilde{s}^2 = s^2 - (1 - \beta) \sigma_n^2
\qquad
\hat{f}(x) = \mu(x) + z^{\ast} \tilde{s}(x) \cdot \mathbf{1}[x \text{ residential}]
$$

Outside residential zones the cost is symmetric and the posterior mean is optimal.
