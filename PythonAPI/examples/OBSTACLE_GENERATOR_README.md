# CARLA 障碍物场景生成器

`obstacle_scenario_generator.py` 是一个可独立预览、也可嵌入数据采集脚本的场景模块。它不依赖固定地图坐标，而是以主车所在车道前方的 waypoint 为参考生成障碍物。

## 场景类型

| 名称 | 内容 | 主要随机变量 |
|---|---|---|
| `clear_control` | 无障碍对照 | 地图、路线、天气、种子 |
| `oblique_vehicle` | 单辆斜停或故障车 | 距离、横向偏移、偏航角、车型 |
| `multi_vehicle_roadblock` | 两至三辆车形成封堵 | 车辆数量、横向排列、纵向扰动、角度 |
| `irregular_barriers` | 非规则锥桶或路障边界 | 数量、间距、缺口、角度、位置抖动 |
| `debris_cluster` | 散落物或货物簇 | 道具类型、数量、位置、任意朝向 |
| `occluded_pedestrian` | 车辆遮挡的横穿行人 | 遮挡侧、行人速度、出现位置 |
| `mixed_compound` | 故障车、道具和行人复合场景 | 上述变量组合 |

生成器提供 `easy`、`medium`、`hard` 三档难度。生成数量如下：

| 难度 | 静态道具 | 障碍车辆 | 横穿行人 |
|---|---:|---:|---:|
| `easy` | 6–9 | 1–2 | 1–2 |
| `medium` | 12–18 | 2–4 | 3–5 |
| `hard` | 20–30 | 3–5 | 5–8 |

生成器会从锚点车道向左右遍历同向 `Driving` 车道，按照整段道路宽度分布对象，并允许在路缘附近增加少量边界扰动。因此多车道路段不会只使用主车所在的第一车道。独立预览入口还会优先选择同向车道数较多的出生点。

路障布局会在 `free_scatter`、`two_clusters`、`broken_chicane` 和 `diagonal_scatter` 之间随机选择，而不是沿车道中心整齐排列。

## 设计原则

1. **运行时发现蓝图**：先查询当前 `BlueprintLibrary`，不会假定所有 CARLA 安装都具有相同静态道具。
2. **可复现**：每个场景保存随机种子；相同地图、主车位置、场景参数和种子可复现实验条件。
3. **真值可追溯**：保存 actor ID、blueprint ID、尺寸、出生变换、当前变换、速度和相对主车位置。
4. **失败可见**：静态道具缺失、出生失败和控制失败都会写入 `failures`，不会静默跳过。
5. **安全清理**：只销毁本生成器创建的对象，不会清除场景内其他车辆或传感器。

## CMD 预览命令

先启动 CARLA，再进入示例目录：

```bat
cd /d D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples
```

列出当前 CARLA 包中生成器可识别的蓝图：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe obstacle_scenario_generator.py --list-blueprints
```

预览困难复合障碍物场景：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe obstacle_scenario_generator.py --scenario mixed_compound --difficulty hard --seed 42 --hold-seconds 30 --activation-delay 5
```

预览非规则路障并指定前方距离：

```bat
C:\Users\yangz\anaconda3\envs\carla-test\python.exe obstacle_scenario_generator.py --scenario irregular_barriers --difficulty medium --distance 30 --seed 100
```

预览结束后，场景对象和主车会自动清理。元数据默认保存在：

```text
PythonAPI/examples/output/obstacle_preview/
```

横穿行人默认延迟 3 秒启动，避免脚本启动后行人立刻离开视野。可用 `--activation-delay` 修改延迟。作为模块嵌入采集脚本时，`generate()` 默认只生成行人但不启动移动；相机就绪后调用 `activate_dynamic_actors()`。

## 嵌入采集脚本

```python
from obstacle_scenario_generator import ObstacleScenarioGenerator

generator = ObstacleScenarioGenerator(world, client, seed=episode_seed)
scene = generator.generate(
    ego,
    scenario_type='mixed_compound',
    difficulty='hard',
    distance_m=25.0,
    scenario_id=episode_id,
)

# 同步采集循环中可以随时更新对象相对位置真值。
current_scene = generator.snapshot()

# 相机和传感器就绪后再启动横穿行人。
generator.activate_dynamic_actors()

# episode 结束前保存最后状态。
generator.save_metadata(scene_json_path)
generator.cleanup()
```

建议每个 episode 创建一个生成器场景，并将 `scene['scenario_id']`、`scenario_type`、`difficulty`、`seed` 写入总 manifest。

## 元数据内容

场景 JSON 包含：

```text
schema_version
scenario_id
scenario_type
difficulty
seed
map
ego_actor_id
requested_distance_m
anchor_waypoint
available_blueprint_counts
objects
failures
```

每个对象包含：

```text
actor_id
blueprint_id
obstacle_class
obstacle_behavior
requested_longitudinal_m
requested_lateral_m
requested_yaw_offset_deg
spawn_transform
dimensions_m
current_transform
velocity
speed_mps
relative_to_ego.longitudinal_m
relative_to_ego.lateral_m
relative_to_ego.distance_m
relative_to_ego.bearing_deg
```

## 已知边界

- `debris_cluster` 只能使用当前 CARLA 包实际注册的静态道具。如果没有合适的散落物蓝图，会退回路障类道具，并在对象元数据中记录 `fallback_pool`。
- Python API 不能在打包版 `WindowsNoEditor` 中直接创建任意新网格。真正新增倒树、碎石等模型需要 Unreal/CARLA 源码资源导入流程。
- Traffic Manager 不一定会主动绕开所有 `static.prop`。生成器负责场景和真值；后续规则避障器仍需根据 actor 真值或感知结果自行决策。
- 场景生成成功不等于对象必然出现在每个相机视野中。正式采集前应增加投影可见性检查和 2D/3D 边界框标注。
