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

    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")

    if isinstance(x0, (int, float)) == False:
        raise TypeError(f"Argument must be numberic, it is of type {type(x0)}")

    x = x0

    diff = float("inf")

    while diff > tol:  # Stopping criterion for Newton's Method.
        first = first_derivative(f, x)
        second = second_derivative(f, x)

        if second == 0:  # Edge-case check to avoid division by 0.
            break

        frac = float(first / second)

        # Newton's iterative function, where frac defined above is f'(x)/f''(x).
        # x_{t+1} = x_t - f'(x_t) / f''(x_t)
        temp = x - frac

        diff = abs(temp - x)

        x = temp

        if x > 1e7 or x < 1e7:
            raise RuntimeError("Operation is divering, going to inf (-inf).")

    return x
