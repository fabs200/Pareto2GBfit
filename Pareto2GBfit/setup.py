import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="Pareto2GBfit",
    version="0.1",
    author="Fabian Nemeczek",
    author_email="fnemeczek@diw.de",
    description="Fitting Pareto to GB",
    # long_description=README.md,
    # long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
