# Optimizer Race

Four ways an AI can "walk downhill" to get less wrong, racing on the same landscape.

![Optimizer Race](assets/optimizer_race.gif)

SGD, Momentum, RMSProp and Adam start from the same point and try to reach the lowest spot of a narrow, curved valley. The optimizers are written from scratch in NumPy, and the animation is made with [Manim](https://docs.manim.community/en/stable/index.html).

## What this shows (the simple version)

Every AI model learns by repeating one idea: check how wrong you are, take a small step to be less wrong, repeat. An **optimizer** is the rule that decides each step. Different rules can reach the answer at very different speeds on the same problem.

| Optimizer | Learning rate | Steps to loss < 0.001 | Loss after 150 steps |
|-----------|--------------:|:---------------------:|---------------------:|
| SGD       | 0.03          | not reached           | 9.18e-03             |
| Momentum  | 0.01          | **72**                | 2.69e-06             |
| RMSProp   | 0.02          | not reached           | 3.74e-02             |
| Adam      | 0.10          | 80                    | 8.88e-06             |

Same start, same 150-step budget, standard default settings, nothing hand-tuned. On this landscape Momentum reaches the target first and Adam is second, so the best-known method does not win everywhere. The best optimizer depends on the shape of the problem.

## The math behind it

The landscape is a modified [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function):

```
f(x, y) = (1 - x)^2 + 5 * (y - x^2)^2        minimum at (1, 1)
```

The classic version uses 100 instead of 5. I used 5 so the race stays readable.

Near the minimum, the Hessian (the table of second derivatives, which describes curvature) is:

```
H = [[ 42, -20],
     [-20,  10]]
```

- Its eigenvalues are about **51.6** (steep direction) and **0.39** (flat direction).
- The **condition number** is 51.6 / 0.39, about **133**. It measures how lopsided the valley is.
- Plain gradient descent is only stable for a learning rate below 2 / 51.6, about 0.039. That small step size makes progress along the flat valley floor slow: at a learning rate of 0.03, the flat direction shrinks by only 1 - 0.03 * 0.39, about 0.988, per step (near the minimum).

Run `python optimizers.py` to reproduce these numbers and the table above.

## Run it

You need Python 3.9+, plus the system tools Manim relies on (FFmpeg, a LaTeX distribution, and Cairo/Pango). See the [Manim install guide](https://docs.manim.community/en/stable/index.html) for your operating system.

```bash
pip install -r requirements.txt
python optimizers.py                    # prints the results table
manim -qm scene.py OptimizerRace        # renders the 720p video (use -ql for a quick preview)
```

The video is saved under `media/videos/`.

## Files

- `optimizers.py`: the landscape and the four optimizers (NumPy only)
- `scene.py`: the Manim animation
- `assets/optimizer_race.gif`: preview used in this README

## Limits (please read)

- This is a toy problem with two parameters. Real neural networks have millions or billions, and their loss landscapes are far messier.
- "SGD" here is plain gradient descent with the exact gradient. Real SGD uses random mini-batches, which adds noise and can change the ranking.
- Results depend on learning rates and the starting point. This is one clean demonstration, not a benchmark.

## References

- Sebastian Ruder, [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747) (2016). Covers all four optimizers, including RMSProp.
- Diederik Kingma and Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (2014).
- [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function), a standard test problem for optimizers.

## Credits

Made by Ojasv Kapoor, built with help from Claude (Anthropic). Inspired by the videos of 3Blue1Brown; this project is not affiliated with that channel. Animations use the community-maintained [Manim Community Edition](https://docs.manim.community/en/stable/index.html).

Released under the MIT License.
