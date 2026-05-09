from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'waypoint_follower'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zdeeno',
    maintainer_email='cihalavalentyn@gmail.com',
    description='Buffers waypoints and drives the robot along them via cmd_vel using a NAV2-style pure pursuit controller.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'waypoint_follower_node = waypoint_follower.waypoint_follower_node:main',
        ],
    },
)
