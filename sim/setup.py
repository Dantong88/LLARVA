from setuptools import setup, find_packages

setup(
    name='llarva-sim',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    license=open('LICENSE').read(),
    zip_safe=False,
    description="LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning",
    author='Dantong Niu',
    author_email='niudantong.88@gmail.com',
    url='https://llarva24.github.io/',
    install_requires=[line for line in open('requirements.txt').readlines() if "@" not in line],
    keywords=['Transformer', 'Behavior-Cloning', 'Langauge', 'Robotics', 'Manipulation'],
)