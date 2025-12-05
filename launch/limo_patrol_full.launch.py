from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 지도/파라미터 파일 경로를 밖에서 넘겨받을 수 있게
    map_arg = DeclareLaunchArgument(
        'map',
        default_value='/home/wego/wego_ws/src/LIMO_Smart_Sentinel/maps/my_map.yaml',
        description='Full path to map yaml file'
    )
    params_arg = DeclareLaunchArgument(
        'params',
        default_value='/home/wego/wego_ws/src/LIMO_Smart_Sentinel/config/nav2_params.yaml',
        description='Full path to nav2 params yaml file'
    )

    map_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params')

    return LaunchDescription([
        map_arg,
        params_arg,

        # (1) 화재 센서 노드
        Node(
            package='limo_patrol',
            executable='flame_sensor_node',
            name='flame_sensor_node',
            output='screen',
            parameters=[{
                'raw_topic': '/flame_raw',   # 필요시 수정
                'flame_threshold': 500,
            }]
        ),

        # (2) 주/야간 + 장애물 인식 노드
        Node(
            package='limo_patrol',
            executable='object_perception_node',
            name='object_perception_node',
            output='screen',
            parameters=[{
                'illumination': 80.0,
                'illumination_threshold': 100.0,
                'yolo_topic': '/yolov5/detections',       # YOLO 토픽에 맞게 수정
                'depth_topic': '/camera/depth/image_raw', # Astra depth 토픽에 맞게 수정
                'max_detection_distance': 2.0,
            }]
        ),

        # (3) waypoint 기반 순찰 + 대응 노드
        Node(
            package='limo_patrol',
            executable='waypoint_patrol_node',
            name='waypoint_patrol_node',
            output='screen',
            parameters=[{
                'waypoint_file': '/home/wego/wego_ws/src/limo_patrol/config/waypoints.yaml',
            }]
        ),

        # Nav2 bringup 은 보통 별도 터미널에서:
        # ros2 launch nav2_bringup bringup_launch.py map:=... params:=...
        # 로 띄우는 걸 추천.
        # (원하면 IncludeLaunchDescription 써서 여기 안에 통합도 가능)
    ])
