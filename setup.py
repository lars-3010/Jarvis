from setuptools import setup, find_packages

setup(
    name="jarvis-ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        "transformers",
        "torch",
        # ... other dependencies
    ],
)