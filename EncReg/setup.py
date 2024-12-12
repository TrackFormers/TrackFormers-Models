import setuptools

setuptools.setup(
    name="HEPTransformer",
    version="1.0",
    author="Nadezhda Dobreva",
    author_email="",
    description=".",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.8',
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
    ]
)