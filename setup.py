from setuptools import find_packages,setup


def get_requirements(file_path:str) -> list:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(    
    name='Academics Performance Prediction',
    version='0.1.0',
    author='Tanush', 
    author_email='atanushreddy@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
    