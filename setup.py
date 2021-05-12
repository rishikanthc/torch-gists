import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch gists", # Replace with your own username
    version="0.0.1",
    author="Rishikanth",
    author_email="r3chandr@ucsd.edu",
    description="A collection of models and utilities for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rishikanthc/torch-snippets",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "torch_gists"},
    packages=setuptools.find_packages(where="torch_gists"),
    python_requires=">=3.6",
)