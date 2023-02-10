import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MolRep", 
    version="0.0.1",
    author="Jiahua Rao, Shuangjia Zheng",
    author_email="raojh6@mail2.sysu.edu.cn",
    description="MolRep: Benchmarking Representation Learning Models for Molecular Property Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jh-SYSU/MolRep",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Zero v1.0 Universal",
        "Operating System :: OS Independent",
    ],
)