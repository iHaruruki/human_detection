from setuptools import setup

package_name = 'human_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='MoveNet-based pose estimation node for ROS 2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'movenet_node = human_detection.movenet_node:main',
        ],
    },
)
