from setuptools import setup, find_packages

setup(
    name="jax_tutorial",
    version="1.0.0",
    author="Jason Chen",
    author_email="xin.chen@fftai.com",
    description="Tutorial for JAX learning",
    python_requires=">=3.11",
    install_requires=[
        "jax",
        "scikit-learn",
    ]
)
