from setuptools import setup, find_packages

setup(
    name="missioncontrol",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pandas_ta",
        "pykalman",
        "hmmlearn",
        "plotly",
        "arch",
    ],
    author="Valter Rebelo",
    description="Mission Control Trading System",
    package_data={
        "": ["data/*", "docs/*"]
    },
    include_package_data=True,
)