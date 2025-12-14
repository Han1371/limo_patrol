from setuptools import setup
import os
from glob import glob

package_name = 'limo_patrol'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wego',
    maintainer_email='wego@example.com',
    description='LIMO patrol package: LiDAR obstacle detection + one-shot day/night mode manager',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_perception_lidar_node = limo_patrol.object_perception_lidar_node:main',
            'mode_manager_node = limo_patrol.mode_manager_node:main',
        ],
    },
)
