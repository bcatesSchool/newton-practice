import numpy as np
from scipy.differentiate import hessian, jacobian


def check_if_vector(x):
  return x.ndim == 1


def optimize(f, x0, tolerance=1e-5):
  """Function that implements multivariate Newton's method using scipy.differentiate."""
  x = np.asarray(x0, dtype=np.float64)

  if not check_if_vector(x):
    raise TypeError("Starting point must be a 1-D array.")

  if not callable(f):
    raise TypeError(f"Argument is not a function, it is of type {type(f)}")

  if not np.issubdtype(x.dtype, np.number):
    raise TypeError(
        f"Argument elements must be numeric, but array dtype is {x.dtype}"
    )

  diff = np.full(shape=x.shape, fill_value=float("inf"))

  while np.any(diff > tolerance):
    # Calculate gradient
    grad_vector = jacobian(f, x).df[0]

    # Calculate Hessian matrix
    hessian_matrix = np.squeeze(hessian(f, x).ddf)

    # Solve the system H * step = grad instead of manual explicit inverting
    try:
      step = np.linalg.solve(hessian_matrix, grad_vector)
    except np.linalg.LinAlgError:
      raise RuntimeError(
          "Hessian matrix is singular and cannot be inverted. Optimization failed."
      )

    temp = x - step

    diff = np.abs(temp - x)
    x = temp

    if np.any((x > 1e7) | (x < -1e7)):
      raise RuntimeError("Operation is diverging, going to inf (-inf).")

  return x
