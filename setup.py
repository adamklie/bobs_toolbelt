from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="bobs_toolbelt",
    version="0.0.0",
    author="Adam Klie",
    author_email="aklie@eng.ucsd.edu",
    description="Bioinformatics toolkit",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/bobs_toolbelt",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
