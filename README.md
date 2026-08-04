# SNNObstacleAvoidance — CARLA 数据采集

本分支提供 CARLA 0.9.15 障碍物场景生成、规则式避障、三方向 RGB 数据采集、多天气采集和地图路线扫描脚本。

## 当前版本

- 避障策略：`rule_based_lane_avoidance_v1.3`
- 障碍物生成器：`avoidable_traffic_v3.1`
- 数据格式：`carla_route_obstacle_capture_v1.1`

## 主要功能

- 自动寻找同时包含直行、左转和右转的路线。
- 同步采集前、左、右三个方向的 RGB 图像。
- 生成汽车、摩托车和行人障碍物。
- 仅在存在完整可行绕行轨迹时生成“可绕行交通参与者”场景。
- 沿弯曲道路 waypoint 检测目标车道和安全间隙。
- 排除公交车、卡车等超宽车型，并使用真实包围盒二次校验。
- 合法变道失败并持续受阻后，可低速跨实线进入安全的同向车道。
- 对红灯、行人和路口场景禁止应急违规绕行。
- 输出图像、控制量、避障决策、障碍物真值和事件标签。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `capture_route_with_obstacles.py` | 路线驾驶、三方向拍摄、规则式避障与标签记录 |
| `obstacle_scenario_generator.py` | 汽车、摩托车、行人及复杂障碍物场景生成 |
| `scan_all_maps_for_turn_routes.py` | 扫描地图中包含左/直/右动作的路线 |
| `capture_multi_weather_v4.py` | 多天气、多方向图像采集 |
| `CAPTURE_ROUTE_WITH_OBSTACLES_README.md` | 路线采集脚本说明 |
| `OBSTACLE_GENERATOR_README.md` | 障碍物生成器说明 |

## 环境要求

- Windows 版 CARLA 0.9.15
- 与 CARLA PythonAPI 匹配的 Python 环境
- CARLA 服务端已经启动，默认连接 `127.0.0.1:2000`

本项目脚本按照 CARLA 的 `PythonAPI/examples` 目录结构组织。可以将仓库内容覆盖到 CARLA 安装目录，或者直接把脚本复制到对应的 `PythonAPI/examples` 目录。

## 推荐运行方式

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\capture_route_with_obstacles.py --scenario avoidable_traffic
```

默认配置：三个方向各 100 张、行驶间隔 1 秒、连续静止最多 3 组、堵塞 12 秒清场、应急跨实线等待 8 秒且限速 8 km/h。

关闭应急违规模式：

```bat
--emergency-rule-mode off
```

## 数据说明

采集结果默认写入 `PythonAPI/examples/output/carla_data_v2`，该目录已通过 `.gitignore` 排除。

新增的应急违规标签包括：`rule_violation`、`violation_type`、`violation_reason`、`blocked_duration_s`、`emergency_maneuver`、`emergency_target_speed_kmh` 和 `return_to_lane_success`。

## 注意事项

- 完整 CARLA 安装包、地图资源和数据集不属于本仓库。
- 行人场景始终优先停车礼让。
- 应急模式只允许跨实线进入安全的同向车道，不借用对向车道，也不自动掉头。

## 协作

- hjk
- yzy
