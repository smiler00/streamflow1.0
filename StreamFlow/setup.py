from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="streamflow",
    version="0.1.0",
    author="Ambroise AMETOESSO",
    author_email="ambroiseamteosso@gmail.com",
    description="A Python library for real-time data stream processing with IoT and AI integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ambroise00/StreamFlow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "asyncio",
        "numpy",
        "pandas",
    ],
    extras_require={
        "mqtt": ["paho-mqtt"],
        "api": ["requests", "aiohttp"],
        "viz": ["plotly"],
        "ai": ["torch", "tensorflow", "scikit-learn", "onnx"],
        "redis": ["redis"],
        "dev": ["pytest", "sphinx", "black", "flake8"],
    },
)
