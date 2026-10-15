import os
from setuptools import find_packages, setup

package_name = 'probe_supervisor'

# Funkcja pomocnicza do rekurencyjnego zbierania plików
def generate_data_files(package_name, directory):
    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]
    
    for path, _, files in os.walk(directory):
        if files:
            install_dir = os.path.join('share', package_name, path)
            file_paths = [os.path.join(path, f) for f in files]
            data_files.append((install_dir, file_paths))
            
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=generate_data_files(package_name, 'objects'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smuchars',
    maintainer_email='smuchars@todo.todo',
    description='TODO: Package description',
    license='TODO: License description',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spawner = probe_supervisor.spawner:main',
        ],
    },
)