from setuptools import find_packages, setup

def get_requirements(file_path):
    '''
    Function to get requirements from .txt file
    '''

    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
    return requirements

setup(
    name = 'CBR-DS-Tech-Task',
    version = '0.0.1',
    author = 'Andrey Lysov',
    author_email = 'lysovandrey13@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)