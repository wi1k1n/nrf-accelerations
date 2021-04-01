import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="rd",
    version="0.0.1",
    author="Ilya Mazlov",
    author_email="mazlov.i.a@gmail.com",
    description="Quick saver for Numpy and Torch variables to disk",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/wi1k1n/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)