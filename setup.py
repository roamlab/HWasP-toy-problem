import setuptools

setuptools.setup(name='hwasp_toy_problem',
      version='0.0.1',
      description='The 1D mass-spring toy problem of hardware-software co-optimization using RL',
      author='Tianjian Chen',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=['cloudpickle==1.1.1', 'garage==2019.10.03', 'mass-spring-envs'],
      python_requires='>=3.6'
)