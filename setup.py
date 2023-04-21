from glob import glob
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def read(rel_path):
    import os
    import codecs
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    import os
    suffix = os.environ["VERSION_SUFFIX" ] if "VERSION_SUFFIX" in os.environ else ""
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1] + suffix
    else:
        raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="autoqrels",
    version=get_version("autoqrels/__init__.py"),
    author="Sean MacAvaney",
    author_email="sean.macavaney@glasgow.ac.uk",
    description="a tool for automatically inferring query relevance assessments (qrels)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanmacavaney/autoqrels",
    include_package_data = True,
    packages=setuptools.find_packages(include=['autoqrels', 'autoqrels.*']),
    install_requires=list(open('requirements.txt')),
    classifiers=[],
    python_requires='>=3.6',
)
