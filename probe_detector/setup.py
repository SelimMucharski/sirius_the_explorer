from glob import glob

from setuptools import find_packages, setup

package_name = 'probe_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smuchars',
    maintainer_email='mucharski.selim@gmail.com',
    description='Probe pose estimation (position + axis) from the OAK-D W and OAK-D SR cameras',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'probe_detector = probe_detector.infrastructure.probe_detector_node:main',
            'sim_evaluator = probe_detector.infrastructure.sim_evaluator_node:main',
            'sim_scenario = probe_detector.presentation.sim_scenario:main',
        ],
    },
)
