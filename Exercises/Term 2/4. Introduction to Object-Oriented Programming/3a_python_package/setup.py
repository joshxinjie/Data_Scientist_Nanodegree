from setuptools import setup

setup(name='distributions',
      version='0.1',
      description='Gaussian distributions',
      packages=['distributions'],
      # project cannot be safely installed and run from a zip file
      zip_safe=False)