# Example: least-squares polynomial fitting

TODO(template): replace with the concept document of the project's first
application.

The example application fits a polynomial $p_\theta(x) = \sum_{k=0}^{K}
\theta_k \phi_k(x)$ to noisy observations of $f(x) = \sin(\pi x)$ on
$[-1, 1]$. The basis functions $\phi_k(x) = \sqrt{2k+1}\, P_k(x)$ are
Legendre polynomials scaled to unit second moment under the uniform
distribution, so they are orthonormal on the interval.

The goal is the coefficient vector $\theta$ that minimizes the expected
squared error

$$
L(\theta) = \mathbb{E}_{x \sim U[-1, 1]}\left[ \left( p_\theta(x) - f(x) \right)^2 \right].
$$

Solving the normal equations would give $\theta$ directly, but the example
exists to exercise a stochastic training loop, so it estimates the gradient of
$L$ from minibatches of noisy samples $y_i = f(x_i) + \epsilon_i$ with
$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. The noise adds variance to each
gradient estimate but no bias, because it has zero mean and is independent of
$x_i$. Because the basis is orthonormal, the Hessian of $L$ is the
identity, and one learning rate suits every coefficient; a monomial basis
$x^k$ would give an ill-conditioned Hessian whose high-degree directions
converge far more slowly.

The learning rate follows a warmup and cosine anneal, and the fit is
evaluated on a fixed grid against the noise-free $f$, with PSNR computed
using the known peak-to-peak range of $f$ on the interval.
