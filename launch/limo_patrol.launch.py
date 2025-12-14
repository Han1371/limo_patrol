from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mode_manager = Node(
    package='limo_patrol',
    executable='mode_manager_node',
    name='mode_manager_node',
    output='screen',
    parameters=[
        {'mode_source': 'auto'},
        {'initial_mode': 'NIGHT'},
        {'brightness_topic': '/camera/color/image_raw'},  # <- 이걸로 쓰는 걸 추천
        {'brightness_threshold': 50.0},
    ],
)


    object_perception = Node(
        package='limo_patrol',
        executable='object_perception_lidar_node',
        name='object_perception_lidar_node',
        output='screen',
        parameters=[{
            'scan_topic': '/scan',
            'cmd_vel_topic': '/cmd_vel',

            'front_angle_deg': 30.0,
            'min_range_m': 0.12,
            'obstacle_range_m': 0.55,
            'min_hit_ratio': 0.08,
            'stop_publish_hz': 10.0,

            'telegram_enable': True,
            'telegram_bot_token': '',   # 여기에 넣거나 launch arg로 바꿔도 됨
            'telegram_chat_id': '',
            'telegram_prefix': '[LIMO]',
        }]
    )

    return LaunchDescription([
        mode_manager,
        object_perception,
    ])
