from setuptools import find_packages, setup

setup(
    name="llmpruner",
    version="0.1.0",
    description="LLM Pruning",
    license_files=["LICENSE"],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    author="Gaetan Lopez",
    url="https://github.com/gaetanlop/LLMPruner",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.0",
    install_requires=[
        "torch>=1.13.0",
        "transformers",
        "tqdm",
        "datasets"
    ],
)
