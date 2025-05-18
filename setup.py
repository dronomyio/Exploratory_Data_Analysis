from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ford-returns-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Analysis of Ford stock returns based on Exercise 4.11 Q1 from Statistics and Data Analysis for Financial Engineering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ford-returns-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.4.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "scipy>=1.8.0",
        "statsmodels>=0.13.0",
        "streamlit>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "ford-analysis=ford_analysis:main",
        ],
    },
)
