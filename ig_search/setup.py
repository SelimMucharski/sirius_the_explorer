from glob import glob

from setuptools import find_packages, setup

package_name = 'ig_search'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz')),
        ('share/' + package_name + '/config/maps', glob('config/maps/*')),
        ('share/' + package_name + '/config/calib', glob('config/calib/*')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smuchars',
    maintainer_email='mucharski.selim@gmail.com',
    description='Przeszukiwanie terenu w poszukiwaniu próbek (ERC Probing).',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mesh_to_heightmap = ig_search.presentation.mesh_to_heightmap:main',
            'traversability_map = ig_search.presentation.traversability_map:main',
            'search_node = ig_search.infrastructure.search_node:main',
        ],
    },
)
