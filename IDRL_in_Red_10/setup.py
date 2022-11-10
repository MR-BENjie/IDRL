import setuptools

VERSION = '1.1.0'

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IDRL",
    version=VERSION,
    author="Shijie Han",
    author_email="22S003048@stu.hit.edu.cn",
    description="IDRL implementation in Red-10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MR-BENjie/IDRL",
    license='Apache License 2.0',
    keywords=["Red-10","Ambiguous identity", "AI", "Reinforcment Learning", "RL", "Torch", "Poker"],
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'rlcard'
    ],
    requires_python='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
