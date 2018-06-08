import setuptools

setuptools.setup(
    name="hazysim",
    version="0.0.1",
    author="Christopher R. Aberger",
    author_email="craberger@gmail.com",
    description="A PyTorch Simulator for Low-Precision Training",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    packages=setuptools.find_packages(),
)