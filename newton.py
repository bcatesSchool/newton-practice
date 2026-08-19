import numpy as np


def first_derivative(f, x, h=1e-5):
    """
    Returns the first derivative of a function f at a value x.
    """
    # f'(x) = f(x+h) - f(x) / h
    return (f(x + h) - f(x)) / h


def second_derivative(f, x, h=1e-5):
    """
    Returns the second derivative of a function f at a value x.
    """
    # f''(x) = f'(x+h) - f'(x) / h
    second = (first_derivative(f, x + h) - first_derivative(f, x)) / h
    return second


def optimize(f, x0, tol=1e-5):
    """
    Function that implements Newton's method for optimization.
    """
    x = x0

    diff = float("inf")

    while diff > tol:  # Stopping criterion for Newton's Method.
        first = first_derivative(f, x)
        second = second_derivative(f, x)

        if second == 0:  # Edge-case check to avoid division by 0.
            continue

        frac = float(first / second)

        # Newton's iterative function, where frac defined above is f'(x)/f''(x).
        # x_{t+1} = x_t - f'(x_t) / f''(x_t)
        temp = x - frac

        diff = abs(temp - x)

        x = temp

    return x


def main():
    """
    Completes Newton's Method optimization with function and starting point of choice.
    """
    est = optimize(np.cos, 2.5)
    print(est)


if __name__ == "__main__":
    main()
