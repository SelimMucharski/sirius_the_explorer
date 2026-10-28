from glob import glob

from setuptools import find_packages, setup

package_name = 'ig_search_experiments'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smuchars',
    maintainer_email='mucharski.selim@gmail.com',
    description='Narzędzia symulacyjne dla ig_search (lokalizacja, sędzia, epizody).',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_truth_localization = '
            'ig_search_experiments.ground_truth_localization:main',
            'physics_step_test = ig_search_experiments.physics_step_test:main',
            'probe_watch = ig_search_experiments.probe_watch:main',
            'sample_store = ig_search_experiments.sample_store:main',
            'run_series = ig_search_experiments.run_series:main',
            'analyze_series = ig_search_experiments.analyze_series:main',
            'grasp_test = ig_search_experiments.grasp_test:main',
            'calib_collect = ig_search_experiments.calib_collect:main',
            'calib_fit = ig_search_experiments.calib_fit:main',
            'offline_series = ig_search_experiments.offline_series:main',
        ],
    },
)
