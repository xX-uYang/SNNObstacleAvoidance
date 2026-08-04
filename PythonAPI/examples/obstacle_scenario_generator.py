#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
CARLA 0.9.15 可复用障碍物场景生成器。

设计目标：
    1. 不依赖固定地图坐标，始终以主车前方道路 waypoint 为参考。
    2. 不假设某个静态道具必然存在，运行时检查 BlueprintLibrary。
    3. 通过数量、位置、横向偏移、角度和组合关系生成非规则场景。
    4. 保存场景随机种子、对象真值和失败原因，便于数据集复现。
    5. 只销毁本生成器创建的 actor，不清理场景里的其他对象。

可作为模块使用：
    from obstacle_scenario_generator import ObstacleScenarioGenerator

    generator = ObstacleScenarioGenerator(world, client, seed=42)
    scene = generator.generate(ego, 'mixed_compound', 'hard')
    generator.save_metadata('scene.json')
    ... 采集图片和控制量 ...
    generator.cleanup()

也可独立预览：
    C:\Users\yangz\anaconda3\envs\carla-test\python.exe ^
      obstacle_scenario_generator.py --scenario mixed_compound --difficulty hard
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_DIST = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'carla', 'dist'))

try:
    sys.path.append(glob.glob(os.path.join(
        CARLA_DIST,
        'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        )))[0])
except IndexError:
    pass

import carla


SCHEMA_VERSION = 'obstacle_scene_v1.1'
GENERATOR_VERSION = 'avoidable_traffic_v3.1'

SCENARIO_TYPES = (
    'clear_control',
    'avoidable_traffic',
    'oblique_vehicle',
    'multi_vehicle_roadblock',
    'irregular_barriers',
    'debris_cluster',
    'occluded_pedestrian',
    'mixed_compound',
)

DIFFICULTY_CONFIG = {
    'easy': {
        'distance_range': (38.0, 50.0),
        'count_range': (6, 9),
        'vehicle_count_range': (1, 2),
        'walker_count_range': (1, 2),
        'yaw_range': (-12.0, 12.0),
        'lateral_ratio': 0.75,
        'longitudinal_spread_m': 7.0,
        'road_margin_m': 0.3,
        'walker_speed': (0.8, 1.4),
    },
    'medium': {
        'distance_range': (27.0, 40.0),
        'count_range': (12, 18),
        'vehicle_count_range': (2, 4),
        'walker_count_range': (3, 5),
        'yaw_range': (-35.0, 35.0),
        'lateral_ratio': 0.9,
        'longitudinal_spread_m': 12.0,
        'road_margin_m': 0.8,
        'walker_speed': (1.2, 2.2),
    },
    'hard': {
        'distance_range': (18.0, 30.0),
        'count_range': (20, 30),
        'vehicle_count_range': (3, 5),
        'walker_count_range': (5, 8),
        'yaw_range': (-70.0, 70.0),
        'lateral_ratio': 1.0,
        'longitudinal_spread_m': 18.0,
        'road_margin_m': 1.2,
        'walker_speed': (2.0, 3.5),
    },
}

PROP_KEYWORDS = (
    'barrier', 'cone', 'warning', 'construction', 'box', 'crate',
    'barrel', 'container', 'debris', 'garbage', 'trash', 'rubbish',
    'bag', 'table', 'chair'
)

BARRIER_KEYWORDS = (
    'barrier', 'cone', 'warning', 'construction'
)

DEBRIS_KEYWORDS = (
    'box', 'crate', 'barrel', 'container', 'debris', 'garbage',
    'trash', 'rubbish', 'bag', 'table', 'chair'
)

BAD_FOUR_WHEEL_WORDS = (
    'micro', 'carlamotors'
)

OVERSIZED_VEHICLE_WORDS = (
    'fusorosa', 'bus', 'firetruck', 'ambulance', 'sprinter',
    'hgv', 'carlacola'
)

BICYCLE_WORDS = (
    'crossbike', 'diamondback', 'gazelle', 'omafiets'
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def location_dict(location):
    return {
        'x': round(float(location.x), 4),
        'y': round(float(location.y), 4),
        'z': round(float(location.z), 4),
    }


def rotation_dict(rotation):
    return {
        'pitch': round(float(rotation.pitch), 4),
        'yaw': round(float(rotation.yaw), 4),
        'roll': round(float(rotation.roll), 4),
    }


def transform_dict(transform):
    return {
        'location': location_dict(transform.location),
        'rotation': rotation_dict(transform.rotation),
    }


def vector_length(vector):
    return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def dot_2d(ax, ay, bx, by):
    return ax * bx + ay * by


class ObstacleScenarioGenerator:
    """在主车前方生成可复现的静态、动态和复合障碍物场景。"""

    def __init__(self, world, client, seed=None):
        self.world = world
        self.client = client
        self.map = world.get_map()
        self.seed = int(seed if seed is not None else time.time() * 1000) & 0x7fffffff
        self.rng = random.Random(self.seed)
        self.actors = []
        self.dynamic_controls = []
        self.scene = None
        self._ego = None
        self._blueprints = self._inspect_blueprints()

    @property
    def scenario_types(self):
        return SCENARIO_TYPES

    def blueprint_summary(self):
        return {
            key: [bp.id for bp in values]
            for key, values in self._blueprints.items()
        }

    def _inspect_blueprints(self):
        library = self.world.get_blueprint_library()

        vehicles = []
        motorcycles = []
        for blueprint in library.filter('vehicle.*'):
            lower_id = blueprint.id.lower()
            wheel_count = None
            if blueprint.has_attribute('number_of_wheels'):
                try:
                    wheel_count = int(
                        blueprint.get_attribute('number_of_wheels')
                    )
                except (TypeError, ValueError):
                    continue
            if wheel_count == 2:
                if any(word in lower_id for word in BICYCLE_WORDS):
                    continue
                motorcycles.append(blueprint)
            elif wheel_count in (None, 4):
                if any(
                        word in lower_id
                        for word in (
                            BAD_FOUR_WHEEL_WORDS +
                            OVERSIZED_VEHICLE_WORDS
                        )):
                    continue
                vehicles.append(blueprint)

        walkers = list(library.filter('walker.pedestrian.*'))
        props = list(library.filter('static.prop.*'))
        useful_props = [
            bp for bp in props
            if any(word in bp.id.lower() for word in PROP_KEYWORDS)
        ]
        barriers = [
            bp for bp in useful_props
            if any(word in bp.id.lower() for word in BARRIER_KEYWORDS)
        ]
        debris = [
            bp for bp in useful_props
            if any(word in bp.id.lower() for word in DEBRIS_KEYWORDS)
        ]

        # 已有调试脚本确认 CARLA 0.9.15 常见包中存在该路障。
        if not barriers:
            try:
                barriers = [library.find('static.prop.streetbarrier')]
            except RuntimeError:
                barriers = []

        return {
            'vehicles': vehicles,
            'motorcycles': motorcycles,
            'walkers': walkers,
            'props': useful_props,
            'barriers': barriers,
            'debris': debris,
        }

    def generate(self, ego, scenario_type='random', difficulty='medium',
                 distance_m=None, scenario_id=None, activate_dynamic=False,
                 context=None):
        """生成一个场景并返回可直接写入 JSON 的元数据字典。"""
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError('未知难度：%s' % difficulty)
        if scenario_type == 'random':
            scenario_type = self.rng.choice([
                name for name in SCENARIO_TYPES if name != 'clear_control'
            ])
        if scenario_type not in SCENARIO_TYPES:
            raise ValueError('未知场景：%s' % scenario_type)
        if ego is None or not ego.is_alive:
            raise RuntimeError('主车不存在或已经销毁')

        self.cleanup()
        self._ego = ego
        config = DIFFICULTY_CONFIG[difficulty]
        if distance_m is None:
            distance_m = self.rng.uniform(*config['distance_range'])
        if distance_m <= 5.0:
            raise ValueError('障碍物距离必须大于 5 米')

        anchor = self._road_anchor(ego, distance_m)
        road_span = self._drivable_road_span(
            anchor, config['road_margin_m']
        )
        scenario_id = scenario_id or '%s_%s' % (
            scenario_type, uuid.uuid4().hex[:10]
        )
        self.scene = {
            'schema_version': SCHEMA_VERSION,
            'generator_version': GENERATOR_VERSION,
            'scenario_id': scenario_id,
            'scenario_type': scenario_type,
            'difficulty': difficulty,
            'seed': self.seed,
            'generated_at_utc': utc_now(),
            'map': self.map.name,
            'ego_actor_id': int(ego.id),
            'requested_distance_m': round(float(distance_m), 3),
            'anchor_waypoint': {
                'road_id': int(anchor.road_id),
                'section_id': int(anchor.section_id),
                'lane_id': int(anchor.lane_id),
                'lane_width_m': round(float(anchor.lane_width), 3),
                'is_junction': bool(anchor.is_junction),
                'transform': transform_dict(anchor.transform),
            },
            'road_span': road_span,
            'context': dict(context or {}),
            'available_blueprint_counts': {
                key: len(value) for key, value in self._blueprints.items()
            },
            'objects': [],
            'failures': [],
        }

        if scenario_type == 'clear_control':
            return self.snapshot()
        if scenario_type == 'avoidable_traffic':
            self._scene_avoidable_traffic(anchor, config)
        elif scenario_type == 'oblique_vehicle':
            self._scene_oblique_vehicle(anchor, config)
        elif scenario_type == 'multi_vehicle_roadblock':
            self._scene_multi_vehicle(anchor, config)
        elif scenario_type == 'irregular_barriers':
            self._scene_irregular_barriers(anchor, config)
        elif scenario_type == 'debris_cluster':
            self._scene_debris_cluster(anchor, config)
        elif scenario_type == 'occluded_pedestrian':
            self._scene_occluded_pedestrian(anchor, config)
        elif scenario_type == 'mixed_compound':
            self._scene_mixed_compound(anchor, config)

        if not self.actors:
            raise RuntimeError(
                '场景没有成功生成任何 actor；请检查 BlueprintLibrary 和出生位置'
            )
        if activate_dynamic:
            self.activate_dynamic_actors()
        return self.snapshot()

    def _road_anchor(self, ego, distance_m):
        ego_waypoint = self.map.get_waypoint(
            ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if ego_waypoint is None:
            raise RuntimeError('无法把主车位置投影到 Driving waypoint')

        candidates = list(ego_waypoint.next(float(distance_m)))
        if not candidates:
            # 逐段前进可绕过距离过长或道路末端造成的空结果。
            waypoint = ego_waypoint
            remaining = float(distance_m)
            while remaining > 0:
                step = min(5.0, remaining)
                next_points = list(waypoint.next(step))
                if not next_points:
                    break
                waypoint = self.rng.choice(next_points)
                remaining -= step
            if waypoint is ego_waypoint:
                raise RuntimeError('主车前方没有可用 waypoint')
            return waypoint
        lane_counts = [
            self._drivable_road_span(candidate, 0.0)['lane_count']
            for candidate in candidates
        ]
        best_count = max(lane_counts)
        best_candidates = [
            candidate for candidate, count in zip(candidates, lane_counts)
            if count == best_count
        ]
        return self.rng.choice(best_candidates)

    def _drivable_road_span(self, anchor, margin_m=0.0):
        """计算当前行驶方向内所有相邻 Driving 车道的横向范围。"""
        lanes = [{
            'road_id': int(anchor.road_id),
            'lane_id': int(anchor.lane_id),
            'center_lateral_m': 0.0,
            'lane_width_m': float(anchor.lane_width),
        }]

        current = anchor
        center = 0.0
        for _ in range(4):
            adjacent = current.get_left_lane()
            if (adjacent is None or
                    adjacent.lane_type != carla.LaneType.Driving or
                    adjacent.lane_id * anchor.lane_id <= 0):
                break
            center -= (float(current.lane_width) + float(adjacent.lane_width)) / 2.0
            lanes.append({
                'road_id': int(adjacent.road_id),
                'lane_id': int(adjacent.lane_id),
                'center_lateral_m': center,
                'lane_width_m': float(adjacent.lane_width),
            })
            current = adjacent

        current = anchor
        center = 0.0
        for _ in range(4):
            adjacent = current.get_right_lane()
            if (adjacent is None or
                    adjacent.lane_type != carla.LaneType.Driving or
                    adjacent.lane_id * anchor.lane_id <= 0):
                break
            center += (float(current.lane_width) + float(adjacent.lane_width)) / 2.0
            lanes.append({
                'road_id': int(adjacent.road_id),
                'lane_id': int(adjacent.lane_id),
                'center_lateral_m': center,
                'lane_width_m': float(adjacent.lane_width),
            })
            current = adjacent

        minimum = min(
            lane['center_lateral_m'] - lane['lane_width_m'] / 2.0
            for lane in lanes
        ) - float(margin_m)
        maximum = max(
            lane['center_lateral_m'] + lane['lane_width_m'] / 2.0
            for lane in lanes
        ) + float(margin_m)
        lanes.sort(key=lambda item: item['center_lateral_m'])
        return {
            'lane_count': len(lanes),
            'min_lateral_m': round(minimum, 3),
            'max_lateral_m': round(maximum, 3),
            'margin_m': round(float(margin_m), 3),
            'lanes': lanes,
        }

    def _lateral_bounds(self, config):
        span = self.scene['road_span']
        minimum = float(span['min_lateral_m'])
        maximum = float(span['max_lateral_m'])
        center = (minimum + maximum) / 2.0
        half_width = (maximum - minimum) / 2.0 * config['lateral_ratio']
        return center - half_width, center + half_width

    def _local_transform(self, anchor, longitudinal=0.0, lateral=0.0,
                         yaw_offset=0.0, z_offset=0.2):
        base = anchor.transform
        forward = base.get_forward_vector()
        right = base.get_right_vector()
        location = carla.Location(
            x=base.location.x + forward.x * longitudinal + right.x * lateral,
            y=base.location.y + forward.y * longitudinal + right.y * lateral,
            z=base.location.z + z_offset
        )
        rotation = carla.Rotation(
            pitch=0.0,
            yaw=base.rotation.yaw + yaw_offset,
            roll=0.0
        )
        return carla.Transform(location, rotation)

    def _spawn(self, blueprint, transform, category, behavior,
               longitudinal, lateral, yaw_offset, extra=None):
        if blueprint is None:
            self.scene['failures'].append({
                'reason': 'missing_blueprint',
                'category': category,
            })
            return None

        actor = None
        spawned_transform = None
        attempted = []
        spawn_errors = []
        # 对道路表面高度和轻微占位冲突做有限重试。
        for z_delta, xy_jitter in ((0.0, 0.0), (0.35, 0.0), (0.7, 0.15)):
            retry = carla.Transform(
                carla.Location(
                    x=transform.location.x + xy_jitter,
                    y=transform.location.y - xy_jitter,
                    z=transform.location.z + z_delta
                ),
                transform.rotation
            )
            attempted.append(transform_dict(retry))
            try:
                actor = self.world.try_spawn_actor(blueprint, retry)
            except RuntimeError as error:
                spawn_errors.append(str(error))
                actor = None
            if actor is not None:
                spawned_transform = retry
                break

        if actor is None:
            self.scene['failures'].append({
                'reason': 'spawn_failed',
                'blueprint_id': blueprint.id,
                'category': category,
                'attempted_transforms': attempted,
                'errors': spawn_errors,
            })
            return None

        if actor.type_id.startswith('vehicle.'):
            try:
                actor.set_autopilot(False)
                actor.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True
                ))
                actor.set_simulate_physics(False)
            except RuntimeError:
                pass

        self.actors.append(actor)
        object_meta = self._object_metadata(
            actor,
            category,
            behavior,
            longitudinal,
            lateral,
            yaw_offset,
            extra or {},
            spawned_transform
        )
        self.scene['objects'].append(object_meta)
        return actor

    def _discard_spawned_actor(self, actor, reason):
        actor_id = int(actor.id)
        blueprint_id = actor.type_id
        self.actors = [item for item in self.actors if item.id != actor_id]
        self.scene['objects'] = [
            item for item in self.scene['objects']
            if item.get('actor_id') != actor_id
        ]
        try:
            actor.destroy()
        except RuntimeError:
            pass
        self.scene['failures'].append({
            'reason': reason,
            'actor_id': actor_id,
            'blueprint_id': blueprint_id,
        })

    def _spawn_vehicle(self, anchor, longitudinal, lateral, yaw_offset,
                       behavior='parked', extra=None):
        pool = self._blueprints['vehicles']
        if not pool:
            self.scene['failures'].append({
                'reason': 'no_four_wheel_vehicle_blueprints'
            })
            return None
        blueprint = self.rng.choice(pool)
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', 'scenario_obstacle')
        transform = self._local_transform(
            anchor, longitudinal, lateral, yaw_offset, z_offset=0.45
        )
        return self._spawn(
            blueprint, transform, 'vehicle', behavior,
            longitudinal, lateral, yaw_offset, extra
        )

    def _spawn_lane_fitting_vehicle(
            self, anchor, longitudinal, lateral, yaw_offset,
            behavior='parked', extra=None):
        maximum_width = max(1.8, float(anchor.lane_width) - 0.55)
        attempts = min(12, max(4, len(self._blueprints['vehicles']) * 2))
        for _ in range(attempts):
            actor = self._spawn_vehicle(
                anchor, longitudinal, lateral, yaw_offset,
                behavior=behavior, extra=extra
            )
            if actor is None:
                continue
            width = float(actor.bounding_box.extent.y) * 2.0
            if width <= maximum_width:
                return actor
            self._discard_spawned_actor(
                actor, 'vehicle_too_wide_for_avoidable_lane'
            )
        return None

    def _spawn_motorcycle(self, anchor, longitudinal, lateral, yaw_offset,
                          behavior='parked', extra=None):
        pool = self._blueprints['motorcycles']
        if not pool:
            self.scene['failures'].append({
                'reason': 'no_motorcycle_blueprints'
            })
            return None
        blueprint = self.rng.choice(pool)
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', 'scenario_obstacle')
        transform = self._local_transform(
            anchor, longitudinal, lateral, yaw_offset, z_offset=0.35
        )
        return self._spawn(
            blueprint, transform, 'motorcycle', behavior,
            longitudinal, lateral, yaw_offset, extra
        )

    def _spawn_prop(self, anchor, pool_name, longitudinal, lateral,
                    yaw_offset, extra=None):
        pool = self._blueprints.get(pool_name, [])
        fallback = None
        if not pool:
            fallback = 'barriers'
            pool = self._blueprints['barriers']
        if not pool:
            self.scene['failures'].append({
                'reason': 'no_usable_static_prop_blueprints',
                'requested_pool': pool_name,
            })
            return None

        blueprint = self.rng.choice(pool)
        lower_id = blueprint.id.lower()
        if any(word in lower_id for word in BARRIER_KEYWORDS):
            category = 'barrier_cone'
        elif any(word in lower_id for word in DEBRIS_KEYWORDS):
            category = 'debris_cargo'
        else:
            category = 'other_static'
        details = dict(extra or {})
        if fallback:
            details['fallback_pool'] = fallback
        transform = self._local_transform(
            anchor, longitudinal, lateral, yaw_offset, z_offset=0.12
        )
        return self._spawn(
            blueprint, transform, category, 'static',
            longitudinal, lateral, yaw_offset, details
        )

    def _spawn_crossing_walker(self, anchor, longitudinal, start_lateral,
                               speed, hidden_by_actor_id=None):
        pool = self._blueprints['walkers']
        if not pool:
            self.scene['failures'].append({
                'reason': 'no_walker_blueprints'
            })
            return None

        blueprint = self.rng.choice(pool)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        transform = self._local_transform(
            anchor,
            longitudinal=longitudinal,
            lateral=start_lateral,
            yaw_offset=90.0 if start_lateral < 0 else -90.0,
            z_offset=0.55
        )
        extra = {
            'crossing_speed_mps': round(float(speed), 3),
            'hidden_by_actor_id': hidden_by_actor_id,
            'activation': 'deferred',
        }
        walker = self._spawn(
            blueprint, transform, 'pedestrian', 'crossing',
            longitudinal, start_lateral,
            90.0 if start_lateral < 0 else -90.0,
            extra
        )
        if walker is None:
            return None

        right = anchor.transform.get_right_vector()
        sign = -1.0 if start_lateral > 0 else 1.0
        direction = carla.Vector3D(
            x=right.x * sign,
            y=right.y * sign,
            z=0.0
        )
        self.dynamic_controls.append({
            'actor_id': int(walker.id),
            'control': carla.WalkerControl(
                direction=direction,
                speed=float(speed),
                jump=False
            ),
        })
        return walker

    def _spawn_walker_group(self, anchor, config, count,
                            longitudinal_center=1.5, preferred_side=None,
                            hidden_by_actor_id=None):
        """在道路两侧生成多人横穿组；出生失败时换位置补偿。"""
        minimum, maximum = self._lateral_bounds(config)
        spawned = []
        attempts = max(count * 4, 8)
        for attempt in range(attempts):
            if len(spawned) >= count:
                break
            if preferred_side is None:
                side = -1.0 if attempt % 2 == 0 else 1.0
            else:
                side = float(preferred_side)
            start_lateral = minimum if side < 0 else maximum
            start_lateral += self.rng.uniform(-0.35, 0.35)
            walker = self._spawn_crossing_walker(
                anchor,
                longitudinal=longitudinal_center + self.rng.uniform(-4.0, 4.0),
                start_lateral=start_lateral,
                speed=self.rng.uniform(*config['walker_speed']),
                hidden_by_actor_id=hidden_by_actor_id
            )
            if walker is not None:
                spawned.append(walker)
        if len(spawned) < count:
            self.scene['failures'].append({
                'reason': 'walker_group_incomplete',
                'requested': int(count),
                'spawned': len(spawned),
            })
        return spawned

    def activate_dynamic_actors(self):
        """启动延迟生成的横穿行人，便于相机准备好后再触发事件。"""
        actors_by_id = {}
        for actor in self.actors:
            try:
                if actor.is_alive:
                    actors_by_id[int(actor.id)] = actor
            except RuntimeError:
                pass
        activated = 0
        for item in self.dynamic_controls:
            if item.get('activated'):
                continue
            actor = actors_by_id.get(item['actor_id'])
            if actor is None:
                continue
            try:
                actor.apply_control(item['control'])
                item['activated'] = True
                activated += 1
            except RuntimeError as error:
                self.scene['failures'].append({
                    'reason': 'walker_control_failed',
                    'actor_id': item['actor_id'],
                    'error': str(error),
                })
        if self.scene is not None:
            self.scene['dynamic_actors_activated'] = activated
            self.scene['dynamic_actors_activated_at_utc'] = utc_now()
            for obj in self.scene['objects']:
                if obj['actor_id'] in actors_by_id and obj['obstacle_behavior'] == 'crossing':
                    obj['extra']['activation'] = (
                        'active' if any(
                            item['actor_id'] == obj['actor_id'] and item.get('activated')
                            for item in self.dynamic_controls
                        ) else 'deferred'
                    )
        return activated

    def _scene_oblique_vehicle(self, anchor, config):
        minimum, maximum = self._lateral_bounds(config)
        lateral = self.rng.uniform(minimum, maximum)
        yaw = self.rng.uniform(*config['yaw_range'])
        self._spawn_vehicle(
            anchor, 0.0, lateral, yaw,
            extra={'layout_role': 'primary_blocker'}
        )

    def _scene_avoidable_traffic(self, anchor, config):
        """只占主车车道，始终保留一条已验证可用的相邻超车道。"""
        context = self.scene.get('context', {})
        reserved_direction = context.get(
            'reserved_passing_lane_direction', ''
        )
        lane_width = float(anchor.lane_width)
        lateral_jitter = min(0.28, lane_width * 0.08)
        self.scene['layout_style'] = 'single_lane_avoidable_participant'
        self.scene['reserved_passing_lane_direction'] = reserved_direction

        roll = self.rng.random()
        if roll < 0.48:
            actor = self._spawn_lane_fitting_vehicle(
                anchor,
                longitudinal=0.0,
                lateral=self.rng.uniform(-lateral_jitter, lateral_jitter),
                yaw_offset=self.rng.uniform(-6.0, 6.0),
                behavior='stopped_vehicle',
                extra={
                    'layout_role': 'single_lane_primary_blocker',
                    'reserved_passing_lane_direction': reserved_direction,
                }
            )
            if actor is not None:
                return

        if roll < 0.78:
            actor = self._spawn_motorcycle(
                anchor,
                longitudinal=0.0,
                lateral=self.rng.uniform(-lateral_jitter, lateral_jitter),
                yaw_offset=self.rng.uniform(-10.0, 10.0),
                behavior='stopped_motorcycle',
                extra={
                    'layout_role': 'single_lane_primary_blocker',
                    'reserved_passing_lane_direction': reserved_direction,
                }
            )
            if actor is not None:
                return

        # 行人从车道中心走向非超车侧，不进入预留相邻车道。
        start_lateral = 0.12 if reserved_direction == 'right' else -0.12
        walker = self._spawn_crossing_walker(
            anchor,
            longitudinal=0.0,
            start_lateral=start_lateral,
            speed=self.rng.uniform(*config['walker_speed']),
            hidden_by_actor_id=None
        )
        if walker is not None:
            for obj in reversed(self.scene['objects']):
                if obj['actor_id'] == int(walker.id):
                    obj['extra'].update({
                        'layout_role': 'single_lane_crossing_pedestrian',
                        'reserved_passing_lane_direction': reserved_direction,
                    })
                    break
            return

        # 没有可用行人或摩托蓝图时，退回单辆四轮车。
        self._spawn_lane_fitting_vehicle(
            anchor,
            longitudinal=0.0,
            lateral=0.0,
            yaw_offset=0.0,
            behavior='stopped_vehicle_fallback',
            extra={
                'layout_role': 'single_lane_primary_blocker',
                'reserved_passing_lane_direction': reserved_direction,
            }
        )

    def _scene_multi_vehicle(self, anchor, config):
        minimum, maximum = self._lateral_bounds(config)
        lane_centers = [
            float(lane['center_lateral_m'])
            for lane in self.scene['road_span']['lanes']
        ]
        count = self.rng.randint(*config['vehicle_count_range'])
        self.scene['layout_style'] = 'staggered_multi_lane_roadblock'
        for index in range(count):
            if lane_centers:
                lateral = self.rng.choice(lane_centers) + self.rng.uniform(-0.55, 0.55)
            else:
                lateral = self.rng.uniform(minimum, maximum)
            lateral = max(minimum, min(maximum, lateral))
            self._spawn_vehicle(
                anchor,
                longitudinal=self.rng.uniform(-5.0, 5.0),
                lateral=lateral,
                yaw_offset=self.rng.uniform(*config['yaw_range']),
                extra={
                    'layout_role': 'roadblock_vehicle',
                    'layout_index': index,
                }
            )
        if config is not DIFFICULTY_CONFIG['easy']:
            construction_workers = max(
                2, config['walker_count_range'][0] - 1
            )
            self._spawn_walker_group(
                anchor,
                config,
                count=construction_workers,
                longitudinal_center=0.0,
                preferred_side=None,
                hidden_by_actor_id=None
            )

    def _scene_irregular_barriers(self, anchor, config):
        minimum, maximum = self._lateral_bounds(config)
        center = (minimum + maximum) / 2.0
        half_width = (maximum - minimum) / 2.0
        spread = float(config['longitudinal_spread_m'])
        count = self.rng.randint(*config['count_range'])
        style = self.rng.choice((
            'free_scatter',
            'two_clusters',
            'broken_chicane',
            'diagonal_scatter',
        ))
        self.scene['layout_style'] = style

        for index in range(count):
            progress = index / max(1, count - 1)
            if style == 'free_scatter':
                longitudinal = self.rng.uniform(-spread / 2.0, spread / 2.0)
                lateral = self.rng.uniform(minimum, maximum)
            elif style == 'two_clusters':
                cluster = -1.0 if index < count / 2.0 else 1.0
                longitudinal = cluster * spread * 0.22 + self.rng.uniform(-2.8, 2.8)
                lateral = center + cluster * half_width * 0.55 + self.rng.uniform(-1.1, 1.1)
            elif style == 'broken_chicane':
                longitudinal = -spread / 2.0 + progress * spread + self.rng.uniform(-0.8, 0.8)
                phase = progress * math.pi * 2.5 + self.rng.uniform(-0.45, 0.45)
                lateral = center + math.sin(phase) * half_width * 0.72 + self.rng.uniform(-0.5, 0.5)
            else:
                longitudinal = -spread / 2.0 + progress * spread + self.rng.uniform(-1.5, 1.5)
                lateral = minimum + progress * (maximum - minimum) + self.rng.uniform(-1.0, 1.0)

            lateral = max(minimum, min(maximum, lateral))
            self._spawn_prop(
                anchor,
                'barriers',
                longitudinal=longitudinal,
                lateral=lateral,
                yaw_offset=self.rng.uniform(-100.0, 100.0),
                extra={
                    'layout_role': 'irregular_boundary',
                    'layout_index': index,
                    'layout_style': style,
                }
            )

        if config is not DIFFICULTY_CONFIG['easy']:
            construction_workers = max(
                2, config['walker_count_range'][0] - 1
            )
            self._spawn_walker_group(
                anchor,
                config,
                count=construction_workers,
                longitudinal_center=0.0,
                preferred_side=None,
                hidden_by_actor_id=None
            )

    def _scene_debris_cluster(self, anchor, config):
        minimum, maximum = self._lateral_bounds(config)
        spread = float(config['longitudinal_spread_m'])
        count = self.rng.randint(*config['count_range'])
        style = self.rng.choice(('wide_scatter', 'dense_clusters', 'trail'))
        self.scene['layout_style'] = style
        for index in range(count):
            if style == 'dense_clusters':
                cluster = self.rng.choice((-1.0, 1.0))
                longitudinal = cluster * spread * 0.2 + self.rng.gauss(0.0, 1.8)
                lateral_center = minimum * 0.65 if cluster < 0 else maximum * 0.65
                lateral = lateral_center + self.rng.gauss(0.0, 0.9)
            elif style == 'trail':
                progress = index / max(1, count - 1)
                longitudinal = -spread / 2.0 + progress * spread + self.rng.uniform(-1.0, 1.0)
                lateral = self.rng.uniform(minimum, maximum)
            else:
                longitudinal = self.rng.uniform(-spread / 2.0, spread / 2.0)
                lateral = self.rng.uniform(minimum, maximum)
            lateral = max(minimum, min(maximum, lateral))
            self._spawn_prop(
                anchor,
                'debris',
                longitudinal=longitudinal,
                lateral=lateral,
                yaw_offset=self.rng.uniform(-180.0, 180.0),
                extra={
                    'layout_role': 'scattered_object',
                    'layout_index': index,
                    'layout_style': style,
                }
            )

    def _scene_occluded_pedestrian(self, anchor, config):
        minimum, maximum = self._lateral_bounds(config)
        side = self.rng.choice((-1.0, 1.0))
        blocker_count = max(1, min(2, config['vehicle_count_range'][1]))
        blockers = []
        for index in range(blocker_count):
            blocker = self._spawn_vehicle(
                anchor,
                longitudinal=self.rng.uniform(-2.0, 2.0) + index * 3.0,
                lateral=(minimum + 0.6) if side < 0 else (maximum - 0.6),
                yaw_offset=self.rng.uniform(-15.0, 15.0),
                extra={
                    'layout_role': 'pedestrian_occluder',
                    'layout_index': index,
                }
            )
            if blocker:
                blockers.append(blocker)
        hidden_by = int(blockers[0].id) if blockers else None
        self._spawn_walker_group(
            anchor,
            config,
            count=self.rng.randint(*config['walker_count_range']),
            longitudinal_center=2.0,
            preferred_side=side,
            hidden_by_actor_id=hidden_by
        )

    def _scene_mixed_compound(self, anchor, config):
        minimum, maximum = self._lateral_bounds(config)
        spread = float(config['longitudinal_spread_m'])
        vehicle_count = self.rng.randint(*config['vehicle_count_range'])
        blockers = []
        for index in range(vehicle_count):
            blocker = self._spawn_vehicle(
                anchor,
                longitudinal=self.rng.uniform(-spread * 0.32, spread * 0.32),
                lateral=self.rng.uniform(minimum, maximum),
                yaw_offset=self.rng.uniform(*config['yaw_range']),
                extra={
                    'layout_role': 'compound_vehicle',
                    'layout_index': index,
                }
            )
            if blocker:
                blockers.append(blocker)

        prop_count = self.rng.randint(*config['count_range'])
        style = self.rng.choice(('free_scatter', 'two_clusters', 'broken_chicane'))
        self.scene['layout_style'] = 'mixed_%s' % style
        for index in range(prop_count):
            if style == 'two_clusters':
                cluster = self.rng.choice((-1.0, 1.0))
                longitudinal = cluster * spread * 0.22 + self.rng.uniform(-3.0, 3.0)
                lateral = self.rng.uniform(minimum, maximum)
            elif style == 'broken_chicane':
                progress = index / max(1, prop_count - 1)
                longitudinal = -spread / 2.0 + progress * spread + self.rng.uniform(-1.0, 1.0)
                lateral = (minimum + maximum) / 2.0 + math.sin(progress * math.pi * 3.0) * (maximum - minimum) * 0.34 + self.rng.uniform(-0.7, 0.7)
            else:
                longitudinal = self.rng.uniform(-spread / 2.0, spread / 2.0)
                lateral = self.rng.uniform(minimum, maximum)
            lateral = max(minimum, min(maximum, lateral))
            self._spawn_prop(
                anchor,
                self.rng.choice(('barriers', 'debris')),
                longitudinal=longitudinal,
                lateral=lateral,
                yaw_offset=self.rng.uniform(-150.0, 150.0),
                extra={
                    'layout_role': 'compound_support',
                    'layout_index': index,
                    'layout_style': style,
                }
            )

        walker_count = self.rng.randint(*config['walker_count_range'])
        hidden_by = int(blockers[0].id) if blockers else None
        self._spawn_walker_group(
            anchor,
            config,
            count=walker_count,
            longitudinal_center=self.rng.uniform(-1.0, 3.0),
            preferred_side=None,
            hidden_by_actor_id=hidden_by
        )

    def _object_metadata(self, actor, category, behavior, longitudinal,
                         lateral, yaw_offset, extra, spawned_transform):
        bounding_box = actor.bounding_box
        return {
            'actor_id': int(actor.id),
            'blueprint_id': actor.type_id,
            'obstacle_class': category,
            'obstacle_behavior': behavior,
            'requested_longitudinal_m': round(float(longitudinal), 3),
            'requested_lateral_m': round(float(lateral), 3),
            'requested_yaw_offset_deg': round(float(yaw_offset), 3),
            'spawn_transform': transform_dict(spawned_transform),
            'dimensions_m': {
                'length': round(float(bounding_box.extent.x * 2.0), 3),
                'width': round(float(bounding_box.extent.y * 2.0), 3),
                'height': round(float(bounding_box.extent.z * 2.0), 3),
            },
            'extra': extra,
        }

    def snapshot(self):
        """更新对象当前真值，并返回场景元数据。"""
        if self.scene is None:
            return None
        ego_transform = self._ego.get_transform() if self._ego else None
        if ego_transform is not None:
            ego_forward = ego_transform.get_forward_vector()
            ego_right = ego_transform.get_right_vector()
            self.scene['ego_transform'] = transform_dict(ego_transform)

        actors_by_id = {}
        for actor in self.actors:
            try:
                if actor.is_alive:
                    actors_by_id[int(actor.id)] = actor
            except RuntimeError:
                pass

        for item in self.scene['objects']:
            actor = actors_by_id.get(item['actor_id'])
            if actor is None:
                item['alive'] = False
                continue
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            item['alive'] = True
            item['current_transform'] = transform_dict(transform)
            item['speed_mps'] = round(float(vector_length(velocity)), 4)
            item['velocity'] = {
                'x': round(float(velocity.x), 4),
                'y': round(float(velocity.y), 4),
                'z': round(float(velocity.z), 4),
            }
            if ego_transform is not None:
                dx = transform.location.x - ego_transform.location.x
                dy = transform.location.y - ego_transform.location.y
                longitudinal = dot_2d(dx, dy, ego_forward.x, ego_forward.y)
                lateral = dot_2d(dx, dy, ego_right.x, ego_right.y)
                distance = math.sqrt(dx * dx + dy * dy)
                bearing = math.degrees(math.atan2(lateral, longitudinal))
                item['relative_to_ego'] = {
                    'longitudinal_m': round(float(longitudinal), 3),
                    'lateral_m': round(float(lateral), 3),
                    'distance_m': round(float(distance), 3),
                    'bearing_deg': round(float(bearing), 3),
                }
        self.scene['snapshot_utc'] = utc_now()
        return self.scene

    def save_metadata(self, path):
        data = self.snapshot()
        if data is None:
            raise RuntimeError('尚未生成场景，无法保存元数据')
        absolute = os.path.abspath(path)
        parent = os.path.dirname(absolute)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(absolute, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return absolute

    def cleanup(self):
        """只销毁生成器创建的障碍物 actor。"""
        alive_ids = []
        for actor in self.actors:
            try:
                if actor.is_alive:
                    alive_ids.append(int(actor.id))
            except RuntimeError:
                pass
        if alive_ids:
            try:
                self.client.apply_batch_sync([
                    carla.command.DestroyActor(actor_id)
                    for actor_id in alive_ids
                ], False)
            except RuntimeError:
                for actor in self.actors:
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass
        self.actors = []
        self.dynamic_controls = []


def parse_args():
    parser = argparse.ArgumentParser(
        description='在主车前方生成可复现的 CARLA 非规则障碍物场景。'
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument(
        '--scenario', choices=('random',) + SCENARIO_TYPES,
        default='mixed_compound'
    )
    parser.add_argument(
        '--difficulty', choices=tuple(DIFFICULTY_CONFIG), default='medium'
    )
    parser.add_argument('--distance', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        '--hold-seconds', type=float, default=15.0,
        help='独立预览时场景保留时间（默认 15 秒）'
    )
    parser.add_argument(
        '--activation-delay', type=float, default=3.0,
        help='预览时等待多久再启动横穿行人（默认 3 秒）'
    )
    parser.add_argument(
        '--metadata', default=None,
        help='元数据 JSON；默认保存到 examples/output/obstacle_preview'
    )
    parser.add_argument(
        '--list-blueprints', action='store_true',
        help='列出生成器识别到的可用车辆、行人和静态道具'
    )
    args = parser.parse_args()
    if args.distance is not None and args.distance <= 5.0:
        parser.error('--distance 必须大于 5 米')
    if args.hold_seconds < 0:
        parser.error('--hold-seconds 不能小于 0')
    if args.activation_delay < 0:
        parser.error('--activation-delay 不能小于 0')
    return args


def spawn_preview_ego(world, rng, lookahead_distance=30.0):
    library = world.get_blueprint_library()
    blueprints = []
    for blueprint in library.filter('vehicle.*'):
        if any(
                word in blueprint.id.lower()
                for word in BAD_FOUR_WHEEL_WORDS):
            continue
        if blueprint.has_attribute('number_of_wheels'):
            try:
                if int(blueprint.get_attribute('number_of_wheels')) != 4:
                    continue
            except (TypeError, ValueError):
                continue
        blueprints.append(blueprint)
    points = list(world.get_map().get_spawn_points())
    if not blueprints:
        raise RuntimeError('当前 CARLA 包中没有可用的四轮车辆蓝图')
    if not points:
        raise RuntimeError('当前地图没有车辆出生点')
    rng.shuffle(points)

    def same_direction_lane_count(point):
        waypoint = world.get_map().get_waypoint(
            point.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return 0
        future = list(waypoint.next(float(lookahead_distance)))
        candidates = future if future else [waypoint]

        def count_at(candidate):
            count = 1
            for getter_name in ('get_left_lane', 'get_right_lane'):
                current = candidate
                for _ in range(4):
                    adjacent = getattr(current, getter_name)()
                    if (adjacent is None or
                            adjacent.lane_type != carla.LaneType.Driving or
                            adjacent.lane_id * candidate.lane_id <= 0):
                        break
                    count += 1
                    current = adjacent
            return count

        return max(count_at(candidate) for candidate in candidates)

    # 稳定排序前先 shuffle：优先多车道，同时保留同车道数内的随机性。
    points.sort(key=same_direction_lane_count, reverse=True)
    for point in points:
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', 'scenario_preview_ego')
        ego = world.try_spawn_actor(blueprint, point)
        if ego is not None:
            ego.apply_control(carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True
            ))
            return ego
    raise RuntimeError('无法生成预览主车')


def position_spectator(world, ego):
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    spectator_transform = carla.Transform(
        carla.Location(
            x=transform.location.x - forward.x * 12.0,
            y=transform.location.y - forward.y * 12.0,
            z=transform.location.z + 8.0
        ),
        carla.Rotation(
            pitch=-20.0,
            yaw=transform.rotation.yaw,
            roll=0.0
        )
    )
    world.get_spectator().set_transform(spectator_transform)


def wait_preview(world, seconds):
    end_time = time.time() + seconds
    synchronous = bool(world.get_settings().synchronous_mode)
    while time.time() < end_time:
        if synchronous:
            world.tick()
        else:
            try:
                world.wait_for_tick(1.0)
            except RuntimeError:
                pass


def main():
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()

    seed = args.seed if args.seed is not None else int(time.time())
    generator = ObstacleScenarioGenerator(world, client, seed=seed)
    if args.list_blueprints:
        print(json.dumps(
            generator.blueprint_summary(), ensure_ascii=False, indent=2
        ))
        return 0

    ego = None
    metadata_path = args.metadata
    if metadata_path is None:
        output_dir = os.path.join(SCRIPT_DIR, 'output', 'obstacle_preview')
        metadata_path = os.path.join(
            output_dir,
            '%s_%s.json' % (
                datetime.now().strftime('%Y%m%d_%H%M%S'),
                args.scenario
            )
        )

    try:
        preview_distance = args.distance
        if preview_distance is None:
            distance_range = DIFFICULTY_CONFIG[args.difficulty]['distance_range']
            preview_distance = sum(distance_range) / 2.0
        ego = spawn_preview_ego(
            world, random.Random(seed), lookahead_distance=preview_distance
        )
        position_spectator(world, ego)
        scene = generator.generate(
            ego,
            scenario_type=args.scenario,
            difficulty=args.difficulty,
            distance_m=preview_distance
        )
        saved_path = generator.save_metadata(metadata_path)

        print('=' * 68)
        print('障碍物场景生成成功')
        print('  场景：%s' % scene['scenario_type'])
        print('  难度：%s' % scene['difficulty'])
        print('  随机种子：%d' % scene['seed'])
        print('  成功 actor：%d' % len(scene['objects']))
        print('  覆盖同向车道：%d 条' % scene['road_span']['lane_count'])
        print(
            '  横向范围：%.1f 至 %.1f 米' % (
                scene['road_span']['min_lateral_m'],
                scene['road_span']['max_lateral_m']
            )
        )
        print('  失败记录：%d' % len(scene['failures']))
        print('  待启动动态行人：%d' % len(generator.dynamic_controls))
        print('  元数据：%s' % saved_path)
        print('  场景将在 CARLA 窗口中保留 %.1f 秒' % args.hold_seconds)
        print('=' * 68)

        delay = min(args.activation_delay, args.hold_seconds)
        if delay > 0:
            print('  先静态展示 %.1f 秒，随后启动横穿行人...' % delay)
            wait_preview(world, delay)
        activated = generator.activate_dynamic_actors()
        if activated:
            print('  已启动 %d 个横穿行人' % activated)
        wait_preview(world, max(0.0, args.hold_seconds - delay))
        generator.save_metadata(metadata_path)
        return 0
    finally:
        generator.cleanup()
        if ego is not None:
            try:
                if ego.is_alive:
                    ego.destroy()
            except RuntimeError:
                pass


if __name__ == '__main__':
    sys.exit(main())
