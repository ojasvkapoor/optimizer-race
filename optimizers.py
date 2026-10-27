"""
Four optimizers written from scratch in NumPy, plus the test landscape.

Landscape: a modified Rosenbrock function
    f(x, y) = (1 - x)^2 + B * (y - x^2)^2        with B = 5
(the classic version uses B = 100; B = 5 is gentler so the race stays readable).
The minimum is at (1, 1) where f = 0.

Note: "SGD" here is plain gradient descent with the exact gradient. There is no
random mini-batch noise, so this is a clean toy problem, not a model of full
neural-network training.

Run this file to reproduce the results table used in the video:
    python optimizers.py
"""
import numpy as np

B = 5.0
START = (-1.5, 2.0)
STEPS = 150
TARGET = 1e-3

# name -> (label, learning rate): standard-looking defaults, not hand-tuned
CONFIG = {
    "sgd": ("SGD", 0.03),
    "mom": ("Momentum", 0.01),
    "rms": ("RMSProp", 0.02),
    "adam": ("Adam", 0.10),
}


def f(p):
    """Loss (the 'height' of the landscape) at point p = (x, y)."""
    x, y = p
    return (1 - x) ** 2 + B * (y - x ** 2) ** 2


def grad(p):
    """Gradient of f: the direction of steepest uphill."""
    x, y = p
    return np.array([-2 * (1 - x) - 4 * B * x * (y - x ** 2), 2 * B * (y - x ** 2)])


def run(name, lr, steps, start, beta=0.9, rho=0.9, beta1=0.9, beta2=0.999, eps=1e-8):
    """Run one optimizer and return its path as an array of shape (steps + 1, 2)."""
    p = np.array(start, float)
    path = [p.copy()]
    v = np.zeros(2)  # momentum velocity
    s = np.zeros(2)  # running average of squared gradients (RMSProp / Adam)
    m = np.zeros(2)  # running average of gradients (Adam)
    t = 0
    for _ in range(steps):
        g = grad(p)
        t += 1
        if name == "sgd":
            p = p - lr * g
        elif name == "mom":
            v = beta * v + g
            p = p - lr * v
        elif name == "rms":
            s = rho * s + (1 - rho) * g * g
            p = p - lr * g / (np.sqrt(s) + eps)
        elif name == "adam":
            m = beta1 * m + (1 - beta1) * g
            s = beta2 * s + (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** t)  # bias correction
            s_hat = s / (1 - beta2 ** t)
            p = p - lr * m_hat / (np.sqrt(s_hat) + eps)
        else:
            raise ValueError(f"unknown optimizer: {name}")
        path.append(p.copy())
    return np.array(path)


if __name__ == "__main__":
    H = np.array([[2 + 8 * B, -4 * B], [-4 * B, 2 * B]])  # Hessian of f at (1, 1)
    lam_min, lam_max = np.linalg.eigvalsh(H)
    print(f"Hessian at (1,1): eigenvalues {lam_min:.3f} and {lam_max:.3f}")
    print(f"condition number = {lam_max / lam_min:.0f}")
    print(f"SGD is stable only for lr < 2/lambda_max = {2 / lam_max:.4f}\n")
    print(f"{'optimizer':<10} {'lr':>6}  steps to loss < {TARGET:g}   loss after {STEPS} steps")
    for key, (label, lr) in CONFIG.items():
        losses = np.array([f(q) for q in run(key, lr, STEPS, START)])
        hit = int(np.argmax(losses < TARGET)) if (losses < TARGET).any() else None
        print(f"{label:<10} {lr:>6}  {str(hit) if hit is not None else 'not reached':<24}{losses[-1]:.2e}")
