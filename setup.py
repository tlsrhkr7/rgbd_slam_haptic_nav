from glob import glob

from setuptools import setup

package_name = 'slam_mapping'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='신동준',
    maintainer_email='tlsrhkr7@gmail.com',
    description='Live D435i RGB-D + IMU mapping with RTAB-Map.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localization_viewer_node = slam_mapping.localization_viewer_node:main',
            'yolo_person_node = slam_mapping.yolo_person_node:main',
            'goal_sender_node = slam_mapping.goal_sender_node:main',
            'path_to_haptic_node = slam_mapping.path_to_haptic_node:main',
            'serial_haptic_node = slam_mapping.serial_haptic_node:main',
            'depth_mask_node = slam_mapping.depth_mask_node:main',
            'mapping_monitor_node = slam_mapping.mapping_monitor_node:main',
        ],
    },
)
