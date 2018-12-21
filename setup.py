
'''
Installation script
'''

from setuptools import setup


if __name__ == '__main__':
    setup(
        name = 'rlambda',
        version = '1.0.0',
        description = 'Python module that allows to generate lambda functions recursively',
        author = 'Vykstorm',
        author_email = 'victorruizgomezdev@gmail.com',
        python_requires = '>=3.6',
        install_requires = [],
        dependency_links = [],
        package_dir= {'rlambda':'src'},
        packages= ['rlambda'],
        keywords = ['lambda', 'ast']
    )