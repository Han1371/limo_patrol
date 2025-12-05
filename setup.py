from setuptools import setup
import os
from glob import glob

package_name = 'limo_patrol'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='LIMO patrol (day/night + fire detection)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'flame_sensor_node = limo_patrol.flame_sensor_node:main',
            'object_perception_node = limo_patrol.object_perception_node:main',
            'waypoint_patrol_node = limo_patrol.waypoint_patrol_node:main',
        ],
    },
)
