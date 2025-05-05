import jax.numpy as jnp


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


x = jnp.arange(5.0)
print(selu(x))

"""
Pure functions

JAX transformation and compilation are designed to work only on Python functions that are functionally pure:
all the input data is passed through the function parameters, all the results are output through the function results. 
A pure function will always return the same result if invoked with the same inputs.
"""
