# CARLA 三方向路线与障碍物采集

脚本：`capture_route_with_obstacles.py`

## 默认采集内容

- 自动构造一条同时包含 `left`、`straight`、`right` 的路线。
- 主车沿路线自动行驶。
- 前、左、右三个 RGB 相机同步采集，每个方向 100 张，共 300 张。
- 行驶时每 1 秒保存一组三方向图片。
- 车速低于 1 km/h 时，每 5 个拍摄时刻只保存一组，减少红灯或堵塞处的重复图片。
- 默认启动 `mixed_compound / medium` 障碍物场景，每保存 25 组照片刷新一次。
- 每轮场景包含车辆、非规则静态道具和横穿行人；行人默认延迟 2 秒启动。

## CMD 运行命令

先启动 CARLA，然后直接在 CMD 中运行下面这一整行：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\capture_route_with_obstacles.py
```

使用困难障碍物、固定随机种子并写入测试集：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\capture_route_with_obstacles.py --scenario mixed_compound --difficulty hard --seed 20260803 --split test
```

如果路线搜索提示缺少某种转向，可以增加搜索次数：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\capture_route_with_obstacles.py --route-attempts 1000
```

## 输出目录

默认写入：

```text
PythonAPI/examples/output/carla_data_v2/
├── manifest.csv
└── episode_<时间>_<编号>/
    ├── rgb/
    │   ├── straight/       100 张
    │   ├── left/           100 张
    │   └── right/          100 张
    ├── control.csv
    ├── obstacles.jsonl
    ├── events.jsonl
    ├── route.json
    └── metadata.json
```

`control.csv` 和总 `manifest.csv` 每行对应一个同步世界帧，三张图片通过同一个 `sample_id` 绑定。

## 主要字段

团队方案中的字段已经保留：

- 对齐主键：`sample_id`、`episode_id`、`route_id`、`frame_id`、`timestamp_ns`
- 输入路径：`image_path`、`image_path_straight`、`image_path_left`、`image_path_right`、`event_path`
- 标签：`true_label`、`label_id`、`label_horizon_ms`、`road_option`
- 控制真值：`steer`、`throttle`、`brake`、`hand_brake`、`reverse`、`gear`
- 车辆状态：`speed`、`speed_mps`、`speed_kmh`、加速度、角速度和主车位姿
- 场景：`map`、`weather`、`route`、道路/车道编号、路口状态、交通灯状态
- 数据治理：`split`
- 模型回填：`snn_label`、`snn_latency_ms`、`snn_model_version`、ANN 三类概率、`ann_model_version`、`available_mask`
- 障碍物：`scenario_id`、`scenario_type`、`difficulty`、`obstacle_seed`、障碍物数量、行人数量、最近障碍物距离
- 安全事件：累计碰撞数、越线数和 `driving_event_path`

采集阶段没有运行 SNN/ANN，因此模型字段按接口约定保留：`snn_label=missing`、`available_mask=0`，其余模型输出为空，等待离线推理后按 `sample_id` 回填。`event_path` 专门预留给 DVS/SNN 事件输入；当前的碰撞与越线日志写在 `driving_event_path`。

`obstacles.jsonl` 每行与一个 `sample_id` 对应，并保存该帧所有障碍车辆、静态物体和行人的：

- actor ID、blueprint ID、障碍物类别和行为
- 出生位置、当前世界坐标、尺寸、速度
- 相对主车的纵向距离、横向距离、直线距离和方位角

## 常用参数

- `--frames 100`：每个相机的图片数量。
- `--capture-interval 1.0`：行驶拍摄间隔。
- `--stationary-sample-every 5`：静止画面降采样倍数。
- `--weather ClearNoon`：天气。
- `--scenario mixed_compound`：障碍物类型。
- `--difficulty medium`：障碍物难度。
- `--obstacle-refresh 25`：每多少组图片刷新障碍物。
- `--obstacle-activation-delay 2`：延迟多久启动横穿行人。
- `--split train`：episode 所属数据划分。
- `--seed 123`：固定路线和障碍物随机种子。

同一 episode 的所有帧必须保持在同一个 `split`，不能再按单张图片随机拆分，以免连续帧泄漏。

## 规则式主动避障

采集脚本默认启用 `--avoidance-mode rule`。决策器会检测当前车道前方的障碍车辆、行人和 `static.prop`，并执行以下规则：

1. 红灯、黄灯或近距离横穿行人：强制停车。
2. 前方车辆或静态障碍物：检查左右相邻车道。
3. 只有相邻车道为同方向 `Driving` 车道、标线允许变道、不在路口附近，且目标车道前后间隙安全时才变道。
4. 自动生成“驶入相邻车道—超过障碍物—返回原车道—接回原路线”的局部轨迹。
5. 没有安全车道、轨迹无法接回原路线或绕障过程中出现新近距离障碍物时停车。

默认运行已经会启用规则式避障：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\capture_route_with_obstacles.py --map Town01
```

相关参数：

- `--avoidance-mode rule`：规则式主动变道绕障。
- `--avoidance-mode brake_only`：保持旧行为，只停车不绕行。
- `--avoidance-mode off`：关闭额外障碍决策，仅按路线控制器行驶。
- `--avoidance-distance 28`：前方障碍检测距离。
- `--pedestrian-stop-distance 18`：行人强制停车距离。
- `--pass-clearance-distance 12`：超过障碍后再行驶的安全距离。
- `--lane-change-cooldown 5`：一次绕障结束后的变道冷却时间。

`control.csv` 会额外记录：

```text
decision_state
decision_action
decision_reason
detected_obstacle_id / type / class
obstacle_distance_m / longitudinal_m / lateral_m
time_to_collision_s
target_lane_id
lane_change_direction
left_lane_available / right_lane_available
left_lane_safe / right_lane_safe
```

常见 `decision_action` 包括 `keep_lane`、`brake`、`lane_change_left`、`lane_change_right` 和 `execute_detour`。这些字段可用于复盘规则决策，也可作为后续模仿学习的监督标签。

## 切换地图与扫描全部地图

CARLA 已启动时，可以使用自带的 `config.py` 查看并切换地图：

```bat
cd /d D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\util
C:\Users\yangz\anaconda3\envs\carla-test\python.exe config.py --list
C:\Users\yangz\anaconda3\envs\carla-test\python.exe config.py --map Town05
```

也可以让采集脚本在开始前切换地图：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\capture_route_with_obstacles.py --map Town05
```

只列出当前 CARLA 服务器可加载的地图：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\scan_all_maps_for_turn_routes.py --list-only
```

扫描全部地图，寻找包含左转、直行、右转且不少于 300 米的路线：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\scan_all_maps_for_turn_routes.py --maps all --min-route-length 300 --attempts 150
```

如果不希望普通版和 `_Opt` 优化版重复扫描，可加 `--exclude-opt`。加 `--open-best` 后，扫描结束会自动加载成功路线最长的地图。扫描结果保存在 `examples/output/map_route_scan/`，其中包含每张地图是否成功、路线长度、动作顺序、起终点坐标和三个路口动作的位置。
