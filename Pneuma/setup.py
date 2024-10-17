from setuptools import find_packages, setup


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


LONG_DESCRIPTION = read_file("README.md")
VERSION = read_file("VERSION")
requirements = read_requirements("pneuma/requirements.txt")

setup(
    name="pneuma_wip",
    version=VERSION,
    author="Author",
    author_email="name@mail.com",
    url="https://github.com/TheDataStation/Pneuma",
    description="Pneuma is a Python package for indexing and querying tables.",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    license="TODO License",
    packages=find_packages(include=["pneuma*"]),
    entry_points={"console_scripts": ["pneuma=pneuma.pneuma:main"]},
    install_requires=requirements,
)
