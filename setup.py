from setuptools import setup, find_packages

setup(
    name="gt-adf",
    version="1.0.0",
    description=(
        "GT-ADF: Graph Transformer-Based Anomaly Detection Framework "
        "for Smart Grid EV Charging Stations"
    ),
    author="Thiruppathy Kesavan V, Gopi R, Md. Jakir Hossen, Danalakshmi D, Emerson Raja Joseph",
    author_email="jakir.hossen@mmu.edu.my",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
        "xai": ["captum>=0.6.0"],
        "notebooks": ["jupyterlab>=4.0.0", "plotly>=5.15.0"],
    },
)
