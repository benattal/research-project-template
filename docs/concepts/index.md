# Concepts

TODO(template): replace this page with the project's concept map.

This directory holds the mathematics of the method and nothing operational.
The concept map is the entry point: it states the central object in a few
paragraphs, shows how each application instantiates it, and gives a reading
order. Each application has one concept document. A derivation long enough to
need figures and its own sections gets a directory
(`concepts/<topic>/index.md` with its assets beside it).

A concept document follows the story in
[writing style](../style/writing.md): the goal, what is wrong with the
existing or obvious approach, and what the method does instead and why that
fixes it. Commands, file paths, and defaults belong in
[implementation](../implementation/architecture.md), not here, except where a
concept names the identifier that implements it.

## Shared construction

TODO(template): define the central object with notation introduced before it
is used. For example, if the method learns a quantity $f$ on a domain
$\mathcal{X}$ from samples $x_i$ with weights $w_i$, state what $f$ is in
terms of the sampling process, why it cannot be computed directly, and how the
method fits it.

$$
\int g(x)\, \mu(\mathrm{d}x) = \mathbb{E}\left[ w\, g(x) \right]
$$

## Applications

Each application chooses a domain and a target quantity; the construction
stays the same.

| Application | Domain | Learned or estimated quantity |
|---|---|---|
| [Example](example.md) | the interval $[-1, 1]$ | polynomial coefficients of a noisy function |

## Reading order

1. Read the shared construction above.
2. Read the application concept you are changing.
3. Use [code architecture](../implementation/architecture.md) and the
   application's implementation page for ownership.
