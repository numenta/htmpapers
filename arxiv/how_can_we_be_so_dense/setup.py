from setuptools import setup, find_packages

with open('requirements.txt') as f:
  requirements = f.read().splitlines()

setup(name="htmpaper.how_can_we_be_so_dense",
      description="'How Can We Be So Dense' source code",
      long_description="Source code and scripts used to recreate all the plots "
                       "and tables used in the paper 'How Can We Be So Dense?'",
      licence="AGPL",
      author="Numenta",
      author_email="help@numenta.org",
      url="https://github.com/numenta/htmpapers",
      packages=find_packages("src"),
      package_dir={"": "src"},
      install_requires=requirements,
      version="1.0",
      )
