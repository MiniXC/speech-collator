from setuptools import setup, find_packages

setup(
    name="speech-collator",
    version="0.1.17",
    description="A collator for speech datasets with different batching strategies and attribute extraction.",
    author="Christoph Minixhofer",
    author_email="christoph.minixhofer@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy<1.24",
        "pyworld>=0.3.3",
        "librosa<0.10",
        "scipy>=1.9.3",
    ],
    python_requires=">=3.6",
    license="MIT",
)
