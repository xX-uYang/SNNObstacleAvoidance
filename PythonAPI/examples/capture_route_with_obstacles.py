#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
CARLA 路线驾驶 + 三方向 RGB + 障碍物真值采集。

默认行为：
    - 自动寻找一条同时包含直行、左转、右转的全局路线；
    - 主车由不依赖额外第三方包的路线控制器驾驶；
    - STRAIGHT / LEFT / RIGHT 三个相机同步拍摄，各 100 张；
    - 行驶时每 1.0 秒保存一组三方向照片；
    - 静止时降采样，且同一次连续停车最多保存 3 组，避免重复画面；
    - 障碍物导致长时间停车时自动清场恢复，并按照片数或模拟时间刷新场景；
    - 输出 control.csv、manifest.csv、obstacles.jsonl、events.jsonl 和 metadata.json。

推荐运行：
    C:\Users\yangz\anaconda3\envs\carla-test\python.exe ^
      D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\capture_route_with_obstacles.py
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import queue
import random
import sys
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_API_DIR = os.path.dirname(SCRIPT_DIR)
CARLA_DIST = os.path.join(PYTHON_API_DIR, 'carla', 'dist')

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

# CARLA 打包版中的 agents 包位于 PythonAPI/carla/agents。
sys.path.append(os.path.join(PYTHON_API_DIR, 'carla'))

import carla
from agents.navigation.local_planner import (
    LocalPlanner,
    RoadOption,
)

from obstacle_scenario_generator import (
    DIFFICULTY_CONFIG,
    SCENARIO_TYPES,
    ObstacleScenarioGenerator,
)


SCHEMA_VERSION = 'carla_route_obstacle_capture_v1.1'
DATASET_VERSION = 'carla_data_v2'
FIXED_DELTA_SECONDS = 0.05
SENSOR_TIMEOUT_SECONDS = 5.0
WARMUP_TICKS = 10
VIEWS = ('STRAIGHT', 'LEFT', 'RIGHT')

OBSTACLE_BLOCKING_REASONS = {
    'pedestrian_priority_stop',
    'new_obstacle_during_detour',
    'brake_only_mode',
    'lane_change_cooldown',
    'lane_change_blocked_near_junction',
    'no_safe_adjacent_lane',
    'detour_path_generation_failed',
}

LABEL_ORDER = ('left', 'straight', 'right')
LABEL_TO_ID = {name: index for index, name in enumerate(LABEL_ORDER)}
REQUIRED_ROUTE_OPTIONS = {
    RoadOption.LEFT,
    RoadOption.STRAIGHT,
    RoadOption.RIGHT,
}

CAMERA_TRANSFORMS = {
    'STRAIGHT': carla.Transform(
        carla.Location(x=1.5, z=1.35),
        carla.Rotation(pitch=-2.0, yaw=0.0)
    ),
    'LEFT': carla.Transform(
        carla.Location(x=0.4, y=-0.8, z=1.35),
        carla.Rotation(pitch=-3.0, yaw=-90.0)
    ),
    'RIGHT': carla.Transform(
        carla.Location(x=0.4, y=0.8, z=1.35),
        carla.Rotation(pitch=-3.0, yaw=90.0)
    ),
}

WEATHERS = {
    'ClearNoon': carla.WeatherParameters.ClearNoon,
    'CloudyNoon': carla.WeatherParameters.CloudyNoon,
    'WetNoon': carla.WeatherParameters.WetNoon,
    'WetSunset': carla.WeatherParameters.WetSunset,
    'ClearSunset': carla.WeatherParameters.ClearSunset,
    'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
    'HardRainNoon': carla.WeatherParameters.HardRainNoon,
}

BAD_VEHICLE_WORDS = (
    'bike', 'motorcycle', 'micro', 'gazelle', 'harley', 'crossbike',
    'diamondback', 'low_rider', 'omafiets', 'carlamotors'
)

CSV_FIELDS = (
    'sample_id', 'episode_id', 'route_id', 'split',
    'frame_id', 'timestamp_ns', 'simulation_time_s',
    'image_path',
    'image_path_straight', 'image_path_left', 'image_path_right',
    'event_path', 'driving_event_path',
    'true_label', 'label_id', 'label_horizon_ms', 'road_option',
    'snn_label', 'snn_latency_ms', 'snn_model_version',
    'ann_prob_left', 'ann_prob_straight', 'ann_prob_right',
    'ann_model_version', 'available_mask',
    'steer', 'throttle', 'brake', 'hand_brake', 'reverse', 'gear',
    'speed', 'speed_mps', 'speed_kmh',
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
    'ego_x', 'ego_y', 'ego_z', 'ego_pitch', 'ego_yaw', 'ego_roll',
    'map', 'weather', 'route',
    'road_id', 'section_id', 'lane_id', 'lane_width_m', 'is_junction',
    'traffic_light_state',
    'decision_state', 'decision_action', 'decision_reason',
    'detected_obstacle_id', 'detected_obstacle_type',
    'detected_obstacle_class', 'obstacle_distance_m',
    'obstacle_longitudinal_m', 'obstacle_lateral_m',
    'time_to_collision_s', 'target_lane_id', 'lane_change_direction',
    'left_lane_available', 'right_lane_available',
    'left_lane_safe', 'right_lane_safe',
    'rule_violation', 'violation_type', 'violation_reason',
    'blocked_duration_s', 'emergency_maneuver',
    'emergency_target_speed_kmh', 'return_to_lane_success',
    'scenario_id', 'scenario_type', 'difficulty', 'obstacle_seed',
    'obstacle_count', 'pedestrian_count', 'nearest_obstacle_distance_m',
    'collision_count', 'lane_invasion_count',
)


class RouteDrivingAgent:
    """不依赖额外包的路线驾驶器，包含可追溯的规则式绕障状态机。"""

    POLICY_VERSION = 'rule_based_lane_avoidance_v1.3'

    def __init__(self, vehicle, world_map, target_speed,
                 avoidance_mode='rule', detection_distance=28.0,
                 pedestrian_stop_distance=18.0,
                 pass_clearance_distance=12.0,
                 lane_change_cooldown=5.0,
                 emergency_rule_mode='cross_solid',
                 emergency_wait_seconds=8.0,
                 emergency_target_speed=8.0):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.world_map = world_map
        self.avoidance_mode = avoidance_mode
        self.detection_distance = float(detection_distance)
        self.pedestrian_stop_distance = float(pedestrian_stop_distance)
        self.pass_clearance_distance = float(pass_clearance_distance)
        self.lane_change_cooldown = float(lane_change_cooldown)
        self.emergency_rule_mode = emergency_rule_mode
        self.emergency_wait_seconds = float(emergency_wait_seconds)
        self.emergency_target_speed = float(emergency_target_speed)
        self.local_planner = LocalPlanner(
            vehicle,
            opt_dict={
                'dt': FIXED_DELTA_SECONDS,
                'target_speed': target_speed,
                'sampling_resolution': 2.0,
                'max_throttle': 0.75,
                'max_brake': 0.6,
                'max_steering': 0.8,
            },
            map_inst=world_map,
        )
        self.original_lane_id = None
        self.target_lane_id = None
        self.detour_active = False
        self.target_lane_reached = False
        self.cooldown_until = 0.0
        self.blocked_since = None
        self.blocked_obstacle_id = None
        self.detour_violation_type = ''
        self.detour_blocked_duration = 0.0
        self.decision = self._empty_decision()

    def set_global_plan(self, route):
        self.local_planner.set_global_plan(
            route, stop_waypoint_creation=True, clean_queue=True
        )

    def get_local_planner(self):
        return self.local_planner

    def done(self):
        return self.local_planner.done()

    def _empty_decision(self):
        return {
            'decision_state': 'CRUISE',
            'decision_action': 'keep_lane',
            'decision_reason': 'no_hazard',
            'detected_obstacle_id': '',
            'detected_obstacle_type': '',
            'detected_obstacle_class': '',
            'obstacle_distance_m': '',
            'obstacle_longitudinal_m': '',
            'obstacle_lateral_m': '',
            'time_to_collision_s': '',
            'target_lane_id': '',
            'lane_change_direction': '',
            'left_lane_available': 0,
            'right_lane_available': 0,
            'left_lane_safe': 0,
            'right_lane_safe': 0,
            'rule_violation': 0,
            'violation_type': '',
            'violation_reason': '',
            'blocked_duration_s': '',
            'emergency_maneuver': '',
            'emergency_target_speed_kmh': '',
            'return_to_lane_success': '',
        }

    def decision_snapshot(self):
        return dict(self.decision)

    def _simulation_time(self):
        return float(self.world.get_snapshot().timestamp.elapsed_seconds)

    @staticmethod
    def _actor_half_width(actor):
        try:
            return max(
                0.2,
                float(actor.bounding_box.extent.y),
                min(1.5, float(actor.bounding_box.extent.x))
            )
        except (AttributeError, RuntimeError):
            return 0.5

    @staticmethod
    def _actor_speed_along(actor, forward):
        try:
            velocity = actor.get_velocity()
            return (
                velocity.x * forward.x + velocity.y * forward.y
            )
        except RuntimeError:
            return 0.0

    @staticmethod
    def _obstacle_class(actor):
        if actor.type_id.startswith('walker.pedestrian.'):
            return 'pedestrian'
        if actor.type_id.startswith('vehicle.'):
            try:
                if int(actor.attributes.get('number_of_wheels', 4)) == 2:
                    return 'motorcycle'
            except (TypeError, ValueError):
                pass
            return 'vehicle'
        if actor.type_id.startswith('static.prop.'):
            return 'static_prop'
        return 'other'

    def _relevant_obstacle_actors(self):
        actors = self.world.get_actors()
        result = []
        for pattern in ('vehicle.*', 'walker.pedestrian.*', 'static.prop.*'):
            for actor in actors.filter(pattern):
                if actor.id != self.vehicle.id:
                    result.append(actor)
        return result

    def _nearest_obstacle_ahead(self, maximum_distance=None):
        maximum_distance = float(
            maximum_distance if maximum_distance is not None
            else self.detection_distance
        )
        ego_transform = self.vehicle.get_transform()
        ego_location = ego_transform.location
        ego_waypoint = self.world_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if ego_waypoint is None:
            return None
        samples = self._lane_samples(ego_waypoint, maximum_distance, 0.0)
        ego_half_width = max(0.8, float(self.vehicle.bounding_box.extent.y))
        ego_speed = speed_mps(self.vehicle)
        nearest = None
        for actor in self._relevant_obstacle_actors():
            try:
                location = actor.get_location()
            except RuntimeError:
                continue
            lane_distance, nearest_waypoint = min(
                samples,
                key=lambda item: item[1].transform.location.distance(location)
            )
            lane_transform = nearest_waypoint.transform
            lane_center = lane_transform.location
            forward = lane_transform.get_forward_vector()
            right = lane_transform.get_right_vector()
            dx = location.x - lane_center.x
            dy = location.y - lane_center.y
            longitudinal = lane_distance + dx * forward.x + dy * forward.y
            lateral = dx * right.x + dy * right.y
            corridor = ego_half_width + self._actor_lateral_extent(
                actor, lane_transform.rotation.yaw
            ) + 0.45
            if not (0.0 < longitudinal <= maximum_distance):
                continue
            if abs(lateral) > corridor:
                continue
            actor_speed = self._actor_speed_along(actor, forward)
            closing_speed = max(0.0, ego_speed - actor_speed)
            ttc = (
                longitudinal / closing_speed
                if closing_speed > 0.1 else None
            )
            item = {
                'actor': actor,
                'actor_id': int(actor.id),
                'actor_type': actor.type_id,
                'obstacle_class': self._obstacle_class(actor),
                'distance_m': ego_location.distance(location),
                'longitudinal_m': longitudinal,
                'lateral_m': lateral,
                'speed_mps': actor_speed,
                'ttc_s': ttc,
            }
            if nearest is None or longitudinal < nearest['longitudinal_m']:
                nearest = item
        return nearest

    @staticmethod
    def _same_direction_driving_lane(origin, candidate):
        return bool(
            candidate is not None and
            candidate.lane_type == carla.LaneType.Driving and
            origin.lane_id * candidate.lane_id > 0
        )

    @staticmethod
    def _lane_change_permitted(waypoint, direction):
        value = str(waypoint.lane_change).lower()
        if 'both' in value or direction in value:
            return True
        marking = (
            waypoint.left_lane_marking
            if direction == 'left'
            else waypoint.right_lane_marking
        )
        marking_value = str(marking.lane_change).lower()
        return 'both' in marking_value or direction in marking_value

    def _adjacent_lane(self, waypoint, direction, require_permission=True):
        if waypoint is None or waypoint.is_junction:
            return None
        if require_permission and not self._lane_change_permitted(
                waypoint, direction):
            return None
        candidate = (
            waypoint.get_left_lane()
            if direction == 'left'
            else waypoint.get_right_lane()
        )
        return candidate if self._same_direction_driving_lane(
            waypoint, candidate
        ) else None

    def _junction_ahead(self, distance_m=15.0):
        plan = list(self.local_planner.get_plan())
        if not plan:
            return False
        travelled = 0.0
        previous = self.vehicle.get_location()
        for waypoint, _ in plan:
            location = waypoint.transform.location
            travelled += previous.distance(location)
            if waypoint.is_junction:
                return True
            if travelled >= distance_m:
                return False
            previous = location
        return False

    @staticmethod
    def _actor_lateral_extent(actor, lane_yaw):
        """计算旋转包围盒在目标车道横向上的半宽。"""
        try:
            actor_yaw = actor.get_transform().rotation.yaw
            yaw_delta = math.radians(normalized_yaw_delta(
                lane_yaw, actor_yaw
            ))
            extent = actor.bounding_box.extent
            return (
                abs(math.sin(yaw_delta)) * float(extent.x) +
                abs(math.cos(yaw_delta)) * float(extent.y)
            )
        except (AttributeError, RuntimeError):
            return 0.5

    def _lane_samples(self, target_waypoint, front_gap, rear_gap):
        """沿弯曲 waypoint 中心线采样，避免用一条切线近似整段车道。"""
        samples = [(0.0, target_waypoint)]
        current = target_waypoint
        travelled = 0.0
        while travelled < front_gap:
            following = self._next_same_lane(current, 2.0)
            if following is None or following.is_junction:
                break
            travelled += current.transform.location.distance(
                following.transform.location
            )
            samples.append((travelled, following))
            current = following

        current = target_waypoint
        travelled = 0.0
        while travelled < rear_gap:
            previous = self._previous_same_lane(current, 2.0)
            if previous is None or previous.is_junction:
                break
            travelled += current.transform.location.distance(
                previous.transform.location
            )
            samples.append((-travelled, previous))
            current = previous
        return samples

    def _lane_clearance(self, target_waypoint, front_gap=38.0,
                        rear_gap=15.0):
        """沿真实弯曲车道检查前后间隙，返回 (是否安全, 最小间隙)。"""
        if target_waypoint is None:
            return False, 0.0
        samples = self._lane_samples(target_waypoint, front_gap, rear_gap)
        minimum_clearance = float('inf')
        ego_speed = speed_mps(self.vehicle)
        for actor in self._relevant_obstacle_actors():
            try:
                location = actor.get_location()
            except RuntimeError:
                continue
            lane_distance, nearest = min(
                samples,
                key=lambda item: item[1].transform.location.distance(location)
            )
            transform = nearest.transform
            center = transform.location
            forward = transform.get_forward_vector()
            right = transform.get_right_vector()
            dx = location.x - center.x
            dy = location.y - center.y
            longitudinal = lane_distance + dx * forward.x + dy * forward.y
            lateral = dx * right.x + dy * right.y
            half_lane = max(1.4, float(nearest.lane_width) / 2.0)
            actor_half_width = self._actor_lateral_extent(
                actor, transform.rotation.yaw
            )
            if abs(lateral) > half_lane + actor_half_width + 0.2:
                continue
            if not (-rear_gap <= longitudinal <= front_gap):
                continue
            clearance = abs(longitudinal)
            minimum_clearance = min(minimum_clearance, clearance)
            actor_speed = self._actor_speed_along(actor, forward)
            if longitudinal >= 0.0:
                obstacle_class = self._obstacle_class(actor)
                if (obstacle_class in ('pedestrian', 'static_prop') or
                        actor_speed < 0.5 or clearance < 15.0):
                    return False, clearance
            else:
                dynamic_rear_gap = max(
                    rear_gap,
                    6.0 + max(0.0, actor_speed - ego_speed) * 2.0
                )
                if abs(longitudinal) < dynamic_rear_gap:
                    return False, clearance
        return True, minimum_clearance

    @staticmethod
    def _next_same_lane(waypoint, distance=2.0):
        candidates = list(waypoint.next(distance))
        if not candidates:
            return None
        same_lane = [
            item for item in candidates
            if item.lane_id == waypoint.lane_id and
            item.lane_type == carla.LaneType.Driving
        ]
        if same_lane:
            return min(
                same_lane,
                key=lambda item: abs(normalized_yaw_delta(
                    waypoint.transform.rotation.yaw,
                    item.transform.rotation.yaw
                ))
            )
        return min(
            candidates,
            key=lambda item: abs(normalized_yaw_delta(
                waypoint.transform.rotation.yaw,
                item.transform.rotation.yaw
            ))
        )

    @staticmethod
    def _previous_same_lane(waypoint, distance=2.0):
        candidates = list(waypoint.previous(distance))
        if not candidates:
            return None
        same_lane = [
            item for item in candidates
            if item.lane_id == waypoint.lane_id and
            item.lane_type == carla.LaneType.Driving
        ]
        pool = same_lane if same_lane else candidates
        return min(
            pool,
            key=lambda item: abs(normalized_yaw_delta(
                waypoint.transform.rotation.yaw,
                item.transform.rotation.yaw
            ))
        )

    def _append_forward(self, plan, waypoint, distance_m, option):
        travelled = 0.0
        current = waypoint
        while travelled < distance_m:
            next_waypoint = self._next_same_lane(current, 2.0)
            if next_waypoint is None or next_waypoint.is_junction:
                return None
            travelled += current.transform.location.distance(
                next_waypoint.transform.location
            )
            plan.append((next_waypoint, option))
            current = next_waypoint
        return current

    def _merge_original_route(self, plan, return_waypoint, original_plan):
        return_location = return_waypoint.transform.location
        return_forward = return_waypoint.transform.get_forward_vector()
        best = None
        for index, (waypoint, _) in enumerate(original_plan):
            if (waypoint.lane_id != return_waypoint.lane_id or
                    waypoint.road_id != return_waypoint.road_id):
                continue
            location = waypoint.transform.location
            dx = location.x - return_location.x
            dy = location.y - return_location.y
            forward_distance = dx * return_forward.x + dy * return_forward.y
            distance = return_location.distance(location)
            if forward_distance < -1.0 or distance > 15.0:
                continue
            if best is None or distance < best[0]:
                best = (distance, index)
        if best is None:
            return False
        plan.extend(original_plan[best[1]:])
        return True

    def _build_detour_plan(self, current_waypoint, direction, obstacle,
                           require_permission=True):
        """生成变道、超过障碍、回原车道并接回全局路线的轨迹。"""
        original_plan = list(self.local_planner.get_plan())
        if not original_plan:
            return None
        plan = [(current_waypoint, RoadOption.LANEFOLLOW)]
        approach = self._append_forward(
            plan, current_waypoint, 4.0, RoadOption.LANEFOLLOW
        )
        if approach is None:
            return None
        target = self._adjacent_lane(
            approach, direction, require_permission=require_permission
        )
        if target is None:
            return None
        change_option = (
            RoadOption.CHANGELANELEFT
            if direction == 'left'
            else RoadOption.CHANGELANERIGHT
        )
        plan.append((target, change_option))

        pass_distance = max(
            24.0,
            float(obstacle['longitudinal_m']) + self.pass_clearance_distance
        )
        target_after_pass = self._append_forward(
            plan, target, pass_distance, RoadOption.LANEFOLLOW
        )
        if target_after_pass is None:
            return None
        return_direction = 'right' if direction == 'left' else 'left'
        return_waypoint = self._adjacent_lane(
            target_after_pass, return_direction,
            require_permission=require_permission
        )
        if return_waypoint is None:
            return None
        return_option = (
            RoadOption.CHANGELANERIGHT
            if direction == 'left'
            else RoadOption.CHANGELANELEFT
        )
        plan.append((return_waypoint, return_option))
        merged = self._merge_original_route(
            plan, return_waypoint, original_plan
        )
        if not merged:
            return None
        return {
            'plan': plan,
            'target_lane_id': int(target.lane_id),
            'original_lane_id': int(current_waypoint.lane_id),
        }

    def find_avoidable_direction(self, obstacle_distance):
        """生成障碍物前验证完整的变道、超越和返回原车道轨迹。"""
        waypoint = self.world_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None or waypoint.is_junction:
            return None
        required_clear_distance = (
            float(obstacle_distance) + self.pass_clearance_distance + 18.0
        )
        if self._junction_ahead(required_clear_distance):
            return None

        simulated_obstacle = {
            'longitudinal_m': min(
                float(obstacle_distance),
                max(10.0, self.detection_distance - 1.0)
            )
        }
        candidates = []
        for direction in ('left', 'right'):
            target = self._adjacent_lane(
                waypoint, direction, require_permission=True
            )
            safe, clearance = self._lane_clearance(
                target,
                front_gap=required_clear_distance,
                rear_gap=18.0
            )
            if not safe:
                continue
            detour = self._build_detour_plan(
                waypoint, direction, simulated_obstacle
            )
            if detour is not None:
                candidates.append((clearance, direction))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _fill_obstacle_decision(self, obstacle):
        if obstacle is None:
            return
        self.decision.update({
            'detected_obstacle_id': obstacle['actor_id'],
            'detected_obstacle_type': obstacle['actor_type'],
            'detected_obstacle_class': obstacle['obstacle_class'],
            'obstacle_distance_m': round(obstacle['distance_m'], 3),
            'obstacle_longitudinal_m': round(
                obstacle['longitudinal_m'], 3
            ),
            'obstacle_lateral_m': round(obstacle['lateral_m'], 3),
            'time_to_collision_s': (
                round(obstacle['ttc_s'], 3)
                if obstacle['ttc_s'] is not None else ''
            ),
        })

    @staticmethod
    def _emergency_stop(control, brake=0.8):
        control.throttle = 0.0
        control.brake = float(brake)
        control.hand_brake = False
        return control

    def _reset_blocked_timer(self):
        self.blocked_since = None
        self.blocked_obstacle_id = None

    def _blocked_duration(self, obstacle, now):
        obstacle_id = obstacle.get('actor_id')
        if self.blocked_obstacle_id != obstacle_id:
            self.blocked_obstacle_id = obstacle_id
            self.blocked_since = now
        if self.blocked_since is None:
            self.blocked_since = now
        return max(0.0, now - self.blocked_since)

    def _limit_emergency_control(self, control):
        speed_kmh = speed_mps(self.vehicle) * 3.6
        if speed_kmh > self.emergency_target_speed:
            control.throttle = 0.0
            control.brake = max(float(control.brake), 0.35)
        else:
            control.throttle = min(float(control.throttle), 0.35)
        return control

    def _fill_violation_decision(self, violation_type, blocked_duration):
        self.decision.update({
            'rule_violation': 1,
            'violation_type': violation_type,
            'violation_reason': 'static_obstacle_blocks_legal_path',
            'blocked_duration_s': round(float(blocked_duration), 3),
            'emergency_maneuver': 'low_speed_detour_and_return',
            'emergency_target_speed_kmh': self.emergency_target_speed,
        })

    def _update_detour_state(self, waypoint, now):
        if not self.detour_active:
            return
        if waypoint.lane_id == self.target_lane_id:
            self.target_lane_reached = True
            self.decision['decision_state'] = 'PASSING'
        elif (self.target_lane_reached and
              waypoint.lane_id == self.original_lane_id):
            completed_violation = self.detour_violation_type
            self.detour_active = False
            self.target_lane_reached = False
            self.cooldown_until = now + self.lane_change_cooldown
            self.decision['decision_state'] = 'RETURNED'
            if completed_violation:
                self._fill_violation_decision(
                    completed_violation, self.detour_blocked_duration
                )
                self.decision['return_to_lane_success'] = 1
            self.detour_violation_type = ''
            self.detour_blocked_duration = 0.0

    def run_step(self):
        control = self.local_planner.run_step()
        now = self._simulation_time()
        waypoint = self.world_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        self.decision = self._empty_decision()
        if waypoint is None:
            self.decision.update({
                'decision_state': 'EMERGENCY_STOP',
                'decision_action': 'brake',
                'decision_reason': 'ego_not_on_driving_lane',
            })
            return self._emergency_stop(control)

        self._update_detour_state(waypoint, now)
        if self.detour_active:
            self.decision.update({
                'decision_state': (
                    'PASSING' if self.target_lane_reached
                    else 'CHANGING_LANE'
                ),
                'decision_action': 'execute_detour',
                'decision_reason': 'detour_plan_active',
                'target_lane_id': self.target_lane_id,
            })
            if self.detour_violation_type:
                self._fill_violation_decision(
                    self.detour_violation_type,
                    self.detour_blocked_duration
                )

        try:
            light_state = self.vehicle.get_traffic_light_state()
            stop_for_light = (
                self.vehicle.is_at_traffic_light() and
                light_state in (
                    carla.TrafficLightState.Red,
                    carla.TrafficLightState.Yellow,
                )
            )
        except RuntimeError:
            stop_for_light = False
        if stop_for_light:
            self._reset_blocked_timer()
            self.decision.update({
                'decision_state': 'STOPPED',
                'decision_action': 'brake',
                'decision_reason': 'red_or_yellow_light',
            })
            return self._emergency_stop(control)

        obstacle = self._nearest_obstacle_ahead()
        self._fill_obstacle_decision(obstacle)
        if obstacle is None or self.avoidance_mode == 'off':
            if not self.detour_active:
                self._reset_blocked_timer()
            return (
                self._limit_emergency_control(control)
                if self.detour_active and self.detour_violation_type
                else control
            )

        if (obstacle['obstacle_class'] == 'pedestrian' and
                obstacle['longitudinal_m'] <= self.pedestrian_stop_distance):
            self._reset_blocked_timer()
            self.decision.update({
                'decision_state': 'STOPPED',
                'decision_action': 'brake',
                'decision_reason': 'pedestrian_priority_stop',
            })
            return self._emergency_stop(control, brake=1.0)

        # 已经在绕障轨迹中时不重复规划；若新障碍过近则优先停车。
        if self.detour_active:
            if obstacle['longitudinal_m'] < 7.0:
                self.decision.update({
                    'decision_state': 'EMERGENCY_STOP',
                    'decision_action': 'brake',
                    'decision_reason': 'new_obstacle_during_detour',
                })
                return self._emergency_stop(control, brake=1.0)
            return (
                self._limit_emergency_control(control)
                if self.detour_violation_type else control
            )

        if self.avoidance_mode == 'brake_only':
            self.decision.update({
                'decision_state': 'STOPPED',
                'decision_action': 'brake',
                'decision_reason': 'brake_only_mode',
            })
            return self._emergency_stop(control)

        if now < self.cooldown_until:
            self.decision.update({
                'decision_state': 'FOLLOWING',
                'decision_action': 'brake',
                'decision_reason': 'lane_change_cooldown',
            })
            return self._emergency_stop(control, brake=0.5)

        if waypoint.is_junction or self._junction_ahead(18.0):
            self.decision.update({
                'decision_state': 'STOPPED',
                'decision_action': 'brake',
                'decision_reason': 'lane_change_blocked_near_junction',
            })
            return self._emergency_stop(control)

        candidates = []
        for direction in ('left', 'right'):
            target = self._adjacent_lane(
                waypoint, direction, require_permission=True
            )
            available = target is not None
            safe, clearance = self._lane_clearance(
                target,
                front_gap=max(
                    38.0,
                    obstacle['longitudinal_m'] +
                    self.pass_clearance_distance + 10.0
                )
            )
            self.decision['%s_lane_available' % direction] = int(available)
            self.decision['%s_lane_safe' % direction] = int(
                available and safe
            )
            if available and safe:
                candidates.append((clearance, direction, target))

        if not candidates:
            blocked_duration = self._blocked_duration(obstacle, now)
            self.decision.update({
                'decision_state': 'WAITING_FOR_GAP',
                'decision_action': 'brake',
                'decision_reason': 'no_safe_adjacent_lane',
                'blocked_duration_s': round(blocked_duration, 3),
            })
            if (self.emergency_rule_mode == 'cross_solid' and
                    blocked_duration >= self.emergency_wait_seconds):
                emergency_candidates = []
                front_gap = max(
                    38.0,
                    obstacle['longitudinal_m'] +
                    self.pass_clearance_distance + 10.0
                )
                for direction in ('left', 'right'):
                    target = self._adjacent_lane(
                        waypoint, direction, require_permission=False
                    )
                    safe, clearance = self._lane_clearance(
                        target, front_gap=front_gap, rear_gap=18.0
                    )
                    if not safe:
                        continue
                    detour = self._build_detour_plan(
                        waypoint, direction, obstacle,
                        require_permission=False
                    )
                    if detour is not None:
                        emergency_candidates.append((
                            clearance, direction, target, detour
                        ))
                if emergency_candidates:
                    emergency_candidates.sort(
                        key=lambda item: item[0], reverse=True
                    )
                    _, direction, target, detour = emergency_candidates[0]
                    self.local_planner.set_global_plan(
                        detour['plan'], stop_waypoint_creation=True,
                        clean_queue=True
                    )
                    self.original_lane_id = detour['original_lane_id']
                    self.target_lane_id = detour['target_lane_id']
                    self.detour_active = True
                    self.target_lane_reached = False
                    self.detour_violation_type = 'cross_solid_line'
                    self.detour_blocked_duration = blocked_duration
                    self.decision.update({
                        'decision_state': 'EMERGENCY_LANE_CHANGE',
                        'decision_action': (
                            'emergency_lane_change_%s' % direction
                        ),
                        'decision_reason': (
                            'legal_lane_change_unavailable_emergency_detour'
                        ),
                        'lane_change_direction': direction,
                        'target_lane_id': int(target.lane_id),
                    })
                    self._fill_violation_decision(
                        self.detour_violation_type, blocked_duration
                    )
                    self._reset_blocked_timer()
                    return self._limit_emergency_control(
                        self.local_planner.run_step()
                    )
            return self._emergency_stop(control)

        # 优先选择间隙更大的相邻车道；无障碍时 clearance 为 inf。
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, direction, target = candidates[0]
        detour = self._build_detour_plan(waypoint, direction, obstacle)
        if detour is None:
            self.decision.update({
                'decision_state': 'WAITING_FOR_GAP',
                'decision_action': 'brake',
                'decision_reason': 'detour_path_generation_failed',
                'lane_change_direction': direction,
                'target_lane_id': int(target.lane_id),
            })
            return self._emergency_stop(control)

        self._reset_blocked_timer()
        self.local_planner.set_global_plan(
            detour['plan'], stop_waypoint_creation=True, clean_queue=True
        )
        self.original_lane_id = detour['original_lane_id']
        self.target_lane_id = detour['target_lane_id']
        self.detour_active = True
        self.target_lane_reached = False
        self.detour_violation_type = ''
        self.detour_blocked_duration = 0.0
        self.decision.update({
            'decision_state': 'CHANGING_LANE',
            'decision_action': 'lane_change_%s' % direction,
            'decision_reason': 'safe_adjacent_lane_selected',
            'lane_change_direction': direction,
            'target_lane_id': self.target_lane_id,
        })
        return self.local_planner.run_step()

def parse_args():
    parser = argparse.ArgumentParser(
        description='驾驶一条包含左/直/右的路线，同步采集三方向 RGB 与障碍物真值。'
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument(
        '--connection-wait', type=float, default=180.0,
        help='等待 CARLA RPC 服务就绪的最长时间，单位秒（默认 180）'
    )
    parser.add_argument(
        '--rpc-attempt-timeout', type=float, default=10.0,
        help='每次连接 CARLA 的超时时间，单位秒（默认 10）'
    )
    parser.add_argument(
        '--map', default=None,
        help='开始采集前加载的地图，例如 Town05；省略则使用当前地图'
    )
    parser.add_argument('--frames', type=int, default=100,
                        help='每个相机保存的图片数（默认 100）')
    parser.add_argument('--capture-interval', type=float, default=1.0,
                        help='行驶时相邻照片的模拟时间间隔，单位秒（默认 1）')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fov', type=float, default=90.0)
    parser.add_argument('--target-speed', type=float, default=18.0,
                        help='路线控制器目标速度，单位 km/h（默认 18）')
    parser.add_argument(
        '--avoidance-mode', choices=('rule', 'brake_only', 'off'),
        default='rule',
        help='避障模式：规则式变道、仅制动或关闭（默认 rule）'
    )
    parser.add_argument(
        '--avoidance-distance', type=float, default=28.0,
        help='开始检测前方障碍物的距离，单位米（默认 28）'
    )
    parser.add_argument(
        '--pedestrian-stop-distance', type=float, default=18.0,
        help='行人在此前方距离内强制停车，单位米（默认 18）'
    )
    parser.add_argument(
        '--pass-clearance-distance', type=float, default=12.0,
        help='超过障碍物后至少再前进的安全距离，单位米（默认 12）'
    )
    parser.add_argument(
        '--lane-change-cooldown', type=float, default=5.0,
        help='完成一次绕障后再次变道的冷却时间，单位秒（默认 5）'
    )
    parser.add_argument(
        '--emergency-rule-mode', choices=('off', 'cross_solid'),
        default='cross_solid',
        help='合法绕障失败后的应急违规模式（默认 cross_solid）'
    )
    parser.add_argument(
        '--emergency-wait-seconds', type=float, default=8.0,
        help='连续受阻多久后允许应急跨实线，模拟秒（默认 8）'
    )
    parser.add_argument(
        '--emergency-target-speed', type=float, default=8.0,
        help='应急违规绕障的目标速度上限，km/h（默认 8）'
    )
    parser.add_argument('--weather', choices=tuple(WEATHERS),
                        default='ClearNoon')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--split', choices=('train', 'val', 'test'),
                        default='train')
    parser.add_argument('--route-attempts', type=int, default=400,
                        help='寻找完整路线时最多尝试的起终点组合数')
    parser.add_argument('--min-route-length', type=float, default=450.0,
                        help='期望路线最短长度，单位米（默认 450）')
    parser.add_argument('--label-horizon-ms', type=int, default=500,
                        help='动作标签向前看的时间窗（默认 500 ms）')
    parser.add_argument('--stationary-speed', type=float, default=1.0,
                        help='低于此速度视为静止，单位 km/h')
    parser.add_argument('--stationary-sample-every', type=int, default=5,
                        help='静止时每几个拍摄时刻保存一组（默认 5）')
    parser.add_argument(
        '--max-stationary-samples', type=int, default=3,
        help='同一次连续停车最多保存多少组；0 表示一组也不保存（默认 3）'
    )
    parser.add_argument(
        '--stationary-reset-distance', type=float, default=1.5,
        help='停车后至少移动多少米才开始新的静止统计，单位米（默认 1.5）'
    )
    parser.add_argument('--max-simulation-seconds', type=float, default=900.0,
                        help='最长模拟时间，防止车辆永久堵塞（默认 900 秒）')
    parser.add_argument(
        '--scenario', choices=('random',) + SCENARIO_TYPES,
        default='avoidable_traffic'
    )
    parser.add_argument(
        '--difficulty', choices=tuple(DIFFICULTY_CONFIG), default='medium'
    )
    parser.add_argument('--obstacle-distance', type=float, default=32.0,
                        help='障碍物场景锚点位于主车前方的距离，单位米')
    parser.add_argument('--obstacle-refresh', type=int, default=25,
                        help='每保存多少组照片刷新障碍物；0 表示不刷新')
    parser.add_argument(
        '--obstacle-refresh-seconds', type=float, default=45.0,
        help='单个障碍物场景最长保留的模拟秒数；0 表示不按时间刷新（默认 45）'
    )
    parser.add_argument(
        '--obstacle-stuck-timeout', type=float, default=12.0,
        help='因障碍物连续停车多久后自动清场，模拟秒；0 表示关闭（默认 12）'
    )
    parser.add_argument(
        '--obstacle-recovery-grace', type=float, default=3.0,
        help='自动清场后无障碍恢复行驶的模拟秒数（默认 3）'
    )
    parser.add_argument('--obstacle-activation-delay', type=float, default=4.0,
                        help='场景生成后延迟启动横穿行人的模拟秒数（默认 4）')
    parser.add_argument('--output', default=None,
                        help='数据集根目录；默认 examples/output/carla_data_v2')
    args = parser.parse_args()

    if args.frames <= 0:
        parser.error('--frames 必须大于 0')
    if args.connection_wait <= 0:
        parser.error('--connection-wait 必须大于 0')
    if args.rpc_attempt_timeout <= 0:
        parser.error('--rpc-attempt-timeout 必须大于 0')
    if args.capture_interval < FIXED_DELTA_SECONDS:
        parser.error('--capture-interval 不能小于 %.2f' % FIXED_DELTA_SECONDS)
    if args.width <= 0 or args.height <= 0:
        parser.error('图像宽高必须大于 0')
    if args.target_speed <= 0:
        parser.error('--target-speed 必须大于 0')
    if args.avoidance_distance <= 5.0:
        parser.error('--avoidance-distance 必须大于 5')
    if args.pedestrian_stop_distance <= 0:
        parser.error('--pedestrian-stop-distance 必须大于 0')
    if args.pass_clearance_distance <= 0:
        parser.error('--pass-clearance-distance 必须大于 0')
    if args.lane_change_cooldown < 0:
        parser.error('--lane-change-cooldown 不能小于 0')
    if args.emergency_wait_seconds < 0:
        parser.error('--emergency-wait-seconds 不能小于 0')
    if args.emergency_target_speed <= 0:
        parser.error('--emergency-target-speed 必须大于 0')
    if args.route_attempts <= 0:
        parser.error('--route-attempts 必须大于 0')
    if args.min_route_length <= 0:
        parser.error('--min-route-length 必须大于 0')
    if args.label_horizon_ms < 0:
        parser.error('--label-horizon-ms 不能小于 0')
    if args.stationary_speed < 0:
        parser.error('--stationary-speed 不能小于 0')
    if args.stationary_sample_every <= 0:
        parser.error('--stationary-sample-every 必须大于 0')
    if args.max_stationary_samples < 0:
        parser.error('--max-stationary-samples 不能小于 0')
    if args.stationary_reset_distance <= 0:
        parser.error('--stationary-reset-distance 必须大于 0')
    if args.max_simulation_seconds <= 0:
        parser.error('--max-simulation-seconds 必须大于 0')
    if args.obstacle_distance <= 5.0:
        parser.error('--obstacle-distance 必须大于 5')
    if args.obstacle_refresh < 0:
        parser.error('--obstacle-refresh 不能小于 0')
    if args.obstacle_refresh_seconds < 0:
        parser.error('--obstacle-refresh-seconds 不能小于 0')
    if args.obstacle_stuck_timeout < 0:
        parser.error('--obstacle-stuck-timeout 不能小于 0')
    if args.obstacle_recovery_grace < 0:
        parser.error('--obstacle-recovery-grace 不能小于 0')
    if args.obstacle_activation_delay < 0:
        parser.error('--obstacle-activation-delay 不能小于 0')
    return args


def connect_to_ready_world(args):
    """等待 RPC 服务可用；地图加载期间短暂断连时自动重试。"""
    client = carla.Client(args.host, args.port)
    deadline = time.time() + args.connection_wait
    attempt = 0
    last_error = None

    while time.time() < deadline:
        attempt += 1
        remaining = max(0.1, deadline - time.time())
        client.set_timeout(min(args.rpc_attempt_timeout, remaining))
        try:
            world = client.get_world()
            print(
                'CARLA RPC 已就绪：%s（第 %d 次连接）' % (
                    world.get_map().name, attempt
                )
            )
            break
        except RuntimeError as error:
            last_error = error
            print(
                '等待 CARLA RPC %s:%d：第 %d 次未就绪，剩余约 %.0f 秒...' % (
                    args.host, args.port, attempt,
                    max(0.0, deadline - time.time())
                )
            )
            time.sleep(min(2.0, max(0.0, deadline - time.time())))
    else:
        raise RuntimeError(
            '等待 CARLA RPC %s:%d 超过 %.0f 秒仍未恢复。'
            '请关闭残留的 CarlaUE4 进程，重新启动 CarlaUE4.exe，'
            '看到地图窗口完全加载后再运行本脚本。最后错误：%s' % (
                args.host, args.port, args.connection_wait, last_error
            )
        )

    if args.map:
        requested_name = args.map.replace('\\', '/').rstrip('/').split('/')[-1]
        current_name = world.get_map().name.replace(
            '\\', '/'
        ).rstrip('/').split('/')[-1]
        if requested_name.lower() == current_name.lower():
            print('当前已经是地图 %s，跳过重复加载' % current_name)
        else:
            print('正在加载地图：%s' % args.map)
            client.set_timeout(max(120.0, args.connection_wait))
            try:
                world = client.load_world(args.map)
            except RuntimeError as error:
                raise RuntimeError(
                    '地图 %s 在 %.0f 秒内没有加载完成。请重启 CARLA 后重试，'
                    '或确认该地图出现在 config.py --list 中。原始错误：%s' % (
                        args.map, max(120.0, args.connection_wait), error
                    )
                )
            print('地图加载完成：%s' % world.get_map().name)

    client.set_timeout(30.0)
    return client, world


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def transform_dict(transform):
    return {
        'location': {
            'x': round(float(transform.location.x), 4),
            'y': round(float(transform.location.y), 4),
            'z': round(float(transform.location.z), 4),
        },
        'rotation': {
            'pitch': round(float(transform.rotation.pitch), 4),
            'yaw': round(float(transform.rotation.yaw), 4),
            'roll': round(float(transform.rotation.roll), 4),
        },
    }


def weather_dict(weather):
    fields = (
        'cloudiness', 'precipitation', 'precipitation_deposits',
        'wind_intensity', 'sun_azimuth_angle', 'sun_altitude_angle',
        'fog_density', 'fog_distance', 'fog_falloff', 'wetness',
        'scattering_intensity', 'mie_scattering_scale',
        'rayleigh_scattering_scale', 'dust_storm'
    )
    result = {}
    for field in fields:
        if hasattr(weather, field):
            result[field] = round(float(getattr(weather, field)), 4)
    return result


def road_option_name(option):
    try:
        return RoadOption(option).name
    except (TypeError, ValueError):
        return 'VOID'


def route_length(route):
    total = 0.0
    for index in range(1, len(route)):
        previous = route[index - 1][0].transform.location
        current = route[index][0].transform.location
        total += previous.distance(current)
    return total


def compressed_maneuvers(route):
    result = []
    for _, option in route:
        if option == RoadOption.LEFT:
            name = 'left'
        elif option == RoadOption.RIGHT:
            name = 'right'
        elif option in (RoadOption.STRAIGHT, RoadOption.LANEFOLLOW):
            # 数据集中的 straight 不只表示“穿过十字路口”，
            # 普通道路上的直行跟车同样属于 straight 类。
            name = 'straight'
        else:
            continue
        if not result or result[-1] != name:
            result.append(name)
    return result


def normalized_yaw_delta(from_yaw, to_yaw):
    """返回 [-180, 180) 范围内的有符号航向角差。"""
    return (float(to_yaw) - float(from_yaw) + 180.0) % 360.0 - 180.0


def connection_option(current_waypoint, candidate_waypoint,
                      threshold_degrees=35.0):
    """根据进入分岔前与驶出路口后的航向角变化判定转向。"""
    probe = candidate_waypoint
    travelled = current_waypoint.transform.location.distance(
        candidate_waypoint.transform.location
    )
    # candidate 常位于路口入口，刚进入时航向变化很小；继续沿该分支
    # 观察到驶出 Junction，才能可靠地区分左、直、右。
    while probe.is_junction and travelled < 35.0:
        successors = list(probe.next(2.0))
        if not successors:
            break
        if len(successors) == 1:
            next_probe = successors[0]
        else:
            # 已进入某条路口连接后，优先保持局部航向连续。
            next_probe = min(
                successors,
                key=lambda item: abs(normalized_yaw_delta(
                    probe.transform.rotation.yaw,
                    item.transform.rotation.yaw
                ))
            )
        travelled += probe.transform.location.distance(
            next_probe.transform.location
        )
        probe = next_probe

    delta = normalized_yaw_delta(
        current_waypoint.transform.rotation.yaw,
        probe.transform.rotation.yaw
    )
    if abs(delta) <= threshold_degrees:
        return RoadOption.STRAIGHT
    if delta > 0.0:
        return RoadOption.LEFT
    return RoadOption.RIGHT


def waypoint_key(waypoint):
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        round(float(waypoint.s), 1),
    )


def same_direction_lane_count(world_map, transform):
    waypoint = world_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        return 0
    count = 1
    for getter_name in ('get_left_lane', 'get_right_lane'):
        current = waypoint
        for _ in range(5):
            adjacent = getattr(current, getter_name)()
            if (adjacent is None or
                    adjacent.lane_type != carla.LaneType.Driving or
                    adjacent.lane_id * waypoint.lane_id <= 0):
                break
            count += 1
            current = adjacent
    return count


def forward_distance_to_junction(world_map, transform, maximum=80.0):
    waypoint = world_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None or waypoint.is_junction:
        return 0.0
    travelled = 0.0
    current = waypoint
    visited = set()
    while travelled < maximum:
        key = waypoint_key(current)
        if key in visited:
            return min(travelled, maximum)
        visited.add(key)
        candidates = list(current.next(2.0))
        if not candidates:
            return min(travelled, maximum)
        same_lane = [
            item for item in candidates
            if item.lane_id == current.lane_id and
            item.lane_type == carla.LaneType.Driving
        ]
        next_waypoint = (same_lane or candidates)[0]
        travelled += current.transform.location.distance(
            next_waypoint.transform.location
        )
        if next_waypoint.is_junction:
            return travelled
        current = next_waypoint
    return maximum


def _coverage_bit(option):
    if option == RoadOption.LEFT:
        return 1
    if option == RoadOption.RIGHT:
        return 2
    if option in (RoadOption.STRAIGHT, RoadOption.LANEFOLLOW):
        return 4
    return 0


def _propagate_junction_options(route):
    """把分岔处的动作标签延续到整个 Junction，方便按时间窗取标签。"""
    if not route:
        return route
    result = [route[0]]
    active = None
    previous_waypoint = route[0][0]
    for waypoint, option in route[1:]:
        if option in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT):
            active = option
        elif (active is not None and
              (previous_waypoint.is_junction or waypoint.is_junction)):
            option = active
        else:
            active = None
            option = RoadOption.LANEFOLLOW
        result.append((waypoint, option))
        previous_waypoint = waypoint
    return result


def _reconstruct_route(final_state, parents, state_waypoints,
                       state_options):
    states = []
    state = final_state
    while state is not None:
        states.append(state)
        state = parents[state]
    states.reverse()
    route = []
    for index, item in enumerate(states):
        option = (
            RoadOption.LANEFOLLOW if index == 0
            else state_options[item]
        )
        route.append((state_waypoints[item], option))
    return _propagate_junction_options(route)


def _extend_route(route, rng, minimum_length):
    """覆盖三类后继续沿路网延长，满足采集所需路线长度。"""
    length_m = route_length(route)
    if not route:
        return route, length_m
    waypoint = route[-1][0]
    maximum_steps = max(200, int(math.ceil(minimum_length / 2.0)) + 300)
    for _ in range(maximum_steps):
        if length_m >= minimum_length:
            break
        candidates = list(waypoint.next(2.0))
        if not candidates:
            break
        if len(candidates) > 1:
            candidate_options = [
                (candidate, connection_option(waypoint, candidate))
                for candidate in candidates
            ]
            next_waypoint, option = rng.choice(candidate_options)
        else:
            next_waypoint = candidates[0]
            option = RoadOption.LANEFOLLOW
        length_m += waypoint.transform.location.distance(
            next_waypoint.transform.location
        )
        route.append((next_waypoint, option))
        waypoint = next_waypoint
    return _propagate_junction_options(route), length_m


def build_route_from_start(world_map, start, rng, min_length,
                           maximum_states=50000):
    """在 waypoint 有向图上搜索覆盖左、直、右的连续路线。"""
    waypoint = world_map.get_waypoint(
        start.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        return None

    # 普通道路上的 LANEFOLLOW 已经满足 straight，因此初始 mask 含 straight。
    start_state = (waypoint_key(waypoint), 4)
    pending = deque([start_state])
    parents = {start_state: None}
    state_waypoints = {start_state: waypoint}
    state_options = {}
    explored = 0

    while pending and explored < maximum_states:
        state = pending.popleft()
        current = state_waypoints[state]
        mask = state[1]
        explored += 1
        candidates = list(current.next(4.0))
        if len(candidates) > 1:
            rng.shuffle(candidates)
        for candidate in candidates:
            option = (
                connection_option(current, candidate)
                if len(candidates) > 1 else RoadOption.LANEFOLLOW
            )
            next_mask = mask | _coverage_bit(option)
            next_state = (waypoint_key(candidate), next_mask)
            if next_state in parents:
                continue
            parents[next_state] = state
            state_waypoints[next_state] = candidate
            state_options[next_state] = option
            if next_mask == 7:
                route = _reconstruct_route(
                    next_state, parents, state_waypoints, state_options
                )
                return _extend_route(route, rng, min_length)
            pending.append(next_state)
    return None


def find_complete_route(world_map, rng, attempts, min_length):
    """从地图出生点中寻找同时包含 LEFT/STRAIGHT/RIGHT 的路线。"""
    spawn_points = list(world_map.get_spawn_points())
    if not spawn_points:
        raise RuntimeError('当前地图的车辆出生点不足，无法规划路线')

    best = None
    best_score = -1.0
    # 先随机打散保证种子可复现，再稳定排序优先多条同向车道。
    # 单车道路段无法执行合法绕障变道，只能停车。
    rng.shuffle(spawn_points)
    spawn_points.sort(
        key=lambda transform: (
            same_direction_lane_count(world_map, transform),
            forward_distance_to_junction(world_map, transform)
        ),
        reverse=True
    )
    start_candidates = [
        spawn_points[index % len(spawn_points)]
        for index in range(min(attempts, max(12, len(spawn_points))))
    ]
    for start in start_candidates:
        try:
            built = build_route_from_start(
                world_map, start, rng, min_length
            )
        except Exception:
            continue
        if built is None:
            continue
        route, length_m = built
        if len(route) < 10:
            continue
        maneuvers = set(compressed_maneuvers(route))
        coverage = len(set(LABEL_ORDER).intersection(maneuvers))
        score = coverage * 100000.0 + length_m
        if score > best_score:
            best = (
                start, route[-1][0].transform, route, length_m
            )
            best_score = score
        if coverage == len(LABEL_ORDER) and length_m >= min_length:
            return start, route[-1][0].transform, route, length_m

    if best is None:
        raise RuntimeError('全局路线规划器没有找到可用路线')
    _, _, best_route, best_length = best
    present = set(compressed_maneuvers(best_route))
    missing = sorted(set(LABEL_ORDER) - present)
    if missing:
        raise RuntimeError(
            '尝试 %d 次仍未找到同时包含左/直/右的路线；缺少：%s。'
            '请增加 --route-attempts 或更换地图。' % (
                attempts, ', '.join(missing)
            )
        )
    print(
        '提示：已找到包含左/直/右的路线，但长度 %.1f 米小于期望 %.1f 米。' % (
            best_length, min_length
        )
    )
    return best


def vehicle_blueprints(world):
    result = []
    for blueprint in world.get_blueprint_library().filter('vehicle.*'):
        if any(word in blueprint.id.lower() for word in BAD_VEHICLE_WORDS):
            continue
        if blueprint.has_attribute('number_of_wheels'):
            try:
                if int(blueprint.get_attribute('number_of_wheels')) != 4:
                    continue
            except (TypeError, ValueError):
                continue
        result.append(blueprint)
    return result


def spawn_ego(world, transform, rng):
    blueprints = vehicle_blueprints(world)
    if not blueprints:
        raise RuntimeError('没有找到可用的四轮车辆 blueprint')
    candidates = list(blueprints)
    rng.shuffle(candidates)
    for blueprint in candidates:
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', 'hero')
        ego = world.try_spawn_actor(blueprint, transform)
        if ego is not None:
            return ego
    # 起点恰好被占用时，尝试将车辆沿 z 轴抬高少许。
    lifted = carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + 0.5
        ),
        transform.rotation
    )
    for blueprint in candidates:
        ego = world.try_spawn_actor(blueprint, lifted)
        if ego is not None:
            return ego
    raise RuntimeError('路线起点被占用，主车生成失败')


def attach_cameras(world, ego, args):
    blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', str(args.width))
    blueprint.set_attribute('image_size_y', str(args.height))
    blueprint.set_attribute('fov', str(args.fov))
    blueprint.set_attribute('sensor_tick', '0.0')
    cameras = {}
    queues = {}
    try:
        for view in VIEWS:
            camera = world.spawn_actor(
                blueprint, CAMERA_TRANSFORMS[view], attach_to=ego
            )
            image_queue = queue.Queue()
            camera.listen(lambda image, q=image_queue: q.put(image))
            cameras[view] = camera
            queues[view] = image_queue
    except Exception:
        for camera in cameras.values():
            try:
                camera.stop()
                camera.destroy()
            except RuntimeError:
                pass
        raise
    return cameras, queues


def attach_event_sensors(world, ego, collision_queue, invasion_queue):
    sensors = []
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)

    def on_collision(event):
        impulse = event.normal_impulse
        collision_queue.put({
            'event_type': 'collision',
            'frame_id': int(event.frame),
            'other_actor_id': int(event.other_actor.id),
            'other_actor_type': event.other_actor.type_id,
            'normal_impulse': {
                'x': round(float(impulse.x), 4),
                'y': round(float(impulse.y), 4),
                'z': round(float(impulse.z), 4),
                'magnitude': round(math.sqrt(
                    impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2
                ), 4),
            },
        })

    collision.listen(on_collision)
    sensors.append(collision)

    invasion_bp = world.get_blueprint_library().find(
        'sensor.other.lane_invasion'
    )
    invasion = world.spawn_actor(invasion_bp, carla.Transform(), attach_to=ego)

    def on_invasion(event):
        invasion_queue.put({
            'event_type': 'lane_invasion',
            'frame_id': int(event.frame),
            'crossed_lane_markings': [
                {
                    'type': str(marking.type),
                    'color': str(marking.color),
                    'lane_change': str(marking.lane_change),
                    'width_m': round(float(marking.width), 4),
                }
                for marking in event.crossed_lane_markings
            ],
        })

    invasion.listen(on_invasion)
    sensors.append(invasion)
    return sensors


def image_for_frame(image_queue, expected_frame):
    deadline = time.time() + SENSOR_TIMEOUT_SECONDS
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise RuntimeError('等待相机世界帧 %d 超时' % expected_frame)
        try:
            image = image_queue.get(timeout=remaining)
        except queue.Empty:
            raise RuntimeError('等待相机世界帧 %d 超时' % expected_frame)
        if image.frame < expected_frame:
            continue
        if image.frame > expected_frame:
            raise RuntimeError(
                '相机跳过世界帧 %d，收到 %d' % (
                    expected_frame, image.frame
                )
            )
        return image


def drain_queue(item_queue):
    while True:
        try:
            item_queue.get_nowait()
        except queue.Empty:
            return


def drain_events(item_queue, event_log, timestamp_ns):
    count = 0
    while True:
        try:
            item = item_queue.get_nowait()
        except queue.Empty:
            return count
        item['timestamp_ns_recorded'] = int(timestamp_ns)
        event_log.append(item)
        count += 1


def speed_mps(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def future_label(agent, horizon_ms, current_speed_mps):
    """从局部规划队列读取未来动作；类别顺序固定为左、直、右。"""
    plan = list(agent.get_local_planner().get_plan())
    if not plan:
        return 'straight', RoadOption.LANEFOLLOW
    sampling_resolution = 2.0
    distance_m = max(
        sampling_resolution,
        current_speed_mps * max(0, horizon_ms) / 1000.0
    )
    last_index = min(len(plan) - 1, int(math.ceil(
        distance_m / sampling_resolution
    )))
    considered = plan[:last_index + 1]
    for _, option in considered:
        if option == RoadOption.LEFT:
            return 'left', option
        if option == RoadOption.RIGHT:
            return 'right', option
        if option == RoadOption.STRAIGHT:
            return 'straight', option
    option = considered[-1][1]
    return 'straight', option


def nearest_obstacle_distance(scene):
    if not scene:
        return ''
    distances = []
    for item in scene.get('objects', []):
        if not item.get('alive', True):
            continue
        relative = item.get('relative_to_ego') or {}
        distance = relative.get('distance_m')
        if distance is not None:
            distances.append(float(distance))
    return round(min(distances), 3) if distances else ''


def pedestrian_count(scene):
    if not scene:
        return 0
    return sum(
        1 for item in scene.get('objects', [])
        if item.get('obstacle_class') == 'pedestrian' and item.get('alive', True)
    )


def make_paths(args, episode_id):
    dataset_root = os.path.abspath(args.output) if args.output else os.path.join(
        SCRIPT_DIR, 'output', DATASET_VERSION
    )
    episode_dir = os.path.join(dataset_root, episode_id)
    rgb_dirs = {}
    for view in VIEWS:
        rgb_dir = os.path.join(episode_dir, 'rgb', view.lower())
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_dirs[view] = rgb_dir
    return {
        'dataset_root': dataset_root,
        'episode_dir': episode_dir,
        'rgb_dirs': rgb_dirs,
        'control_csv': os.path.join(episode_dir, 'control.csv'),
        'manifest_csv': os.path.join(dataset_root, 'manifest.csv'),
        'obstacles_jsonl': os.path.join(episode_dir, 'obstacles.jsonl'),
        'events_jsonl': os.path.join(episode_dir, 'events.jsonl'),
        'route_json': os.path.join(episode_dir, 'route.json'),
        'metadata_json': os.path.join(episode_dir, 'metadata.json'),
    }


def relative_path(path, root):
    return os.path.relpath(path, root).replace(os.sep, '/')


def append_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def stop_and_destroy(client, actors):
    alive_ids = []
    for actor in actors:
        if actor is None:
            continue
        try:
            if actor.type_id.startswith('sensor.'):
                actor.stop()
        except RuntimeError:
            pass
        try:
            if actor.is_alive:
                alive_ids.append(int(actor.id))
        except RuntimeError:
            pass
    if alive_ids:
        try:
            client.apply_batch_sync([
                carla.command.DestroyActor(actor_id)
                for actor_id in alive_ids
            ], False)
        except RuntimeError:
            pass


def build_row(ego, control, waypoint, sample_id, episode_id, route_id,
              frame_id, timestamp_ns, elapsed_seconds, image_paths,
              label, option, args, world_map, scene, decision,
              collision_total, invasion_total):
    transform = ego.get_transform()
    acceleration = ego.get_acceleration()
    angular = ego.get_angular_velocity()
    current_speed_mps = speed_mps(ego)
    traffic_light_state = 'None'
    try:
        if ego.is_at_traffic_light():
            traffic_light_state = str(ego.get_traffic_light_state())
    except RuntimeError:
        traffic_light_state = 'Unknown'

    scene = scene or {}
    objects = scene.get('objects', [])
    return {
        'sample_id': sample_id,
        'episode_id': episode_id,
        'route_id': route_id,
        'split': args.split,
        'frame_id': int(frame_id),
        'timestamp_ns': int(timestamp_ns),
        'simulation_time_s': round(float(elapsed_seconds), 6),
        # 兼容前一版 manifest：image_path 默认指向主前视图。
        'image_path': image_paths['STRAIGHT'],
        'image_path_straight': image_paths['STRAIGHT'],
        'image_path_left': image_paths['LEFT'],
        'image_path_right': image_paths['RIGHT'],
        # 当前脚本不采集 DVS/SNN 事件流，event_path 保持为空；
        # 碰撞和越线事件单独写入 driving_event_path。
        'event_path': '',
        'driving_event_path': relative_path(
            args._paths['events_jsonl'], args._paths['dataset_root']
        ),
        'true_label': label,
        'label_id': LABEL_TO_ID[label],
        'label_horizon_ms': args.label_horizon_ms,
        'road_option': road_option_name(option),
        'snn_label': 'missing',
        'snn_latency_ms': '',
        'snn_model_version': '',
        'ann_prob_left': '',
        'ann_prob_straight': '',
        'ann_prob_right': '',
        'ann_model_version': '',
        'available_mask': 0,
        'steer': round(float(control.steer), 6),
        'throttle': round(float(control.throttle), 6),
        'brake': round(float(control.brake), 6),
        'hand_brake': int(bool(control.hand_brake)),
        'reverse': int(bool(control.reverse)),
        'gear': int(control.gear),
        # speed 字段沿用团队 manifest，单位固定为 km/h。
        'speed': round(current_speed_mps * 3.6, 6),
        'speed_mps': round(current_speed_mps, 6),
        'speed_kmh': round(current_speed_mps * 3.6, 6),
        'acceleration_x': round(float(acceleration.x), 6),
        'acceleration_y': round(float(acceleration.y), 6),
        'acceleration_z': round(float(acceleration.z), 6),
        'angular_velocity_x': round(float(angular.x), 6),
        'angular_velocity_y': round(float(angular.y), 6),
        'angular_velocity_z': round(float(angular.z), 6),
        'ego_x': round(float(transform.location.x), 6),
        'ego_y': round(float(transform.location.y), 6),
        'ego_z': round(float(transform.location.z), 6),
        'ego_pitch': round(float(transform.rotation.pitch), 6),
        'ego_yaw': round(float(transform.rotation.yaw), 6),
        'ego_roll': round(float(transform.rotation.roll), 6),
        'map': world_map.name,
        'weather': args.weather,
        'route': route_id,
        'road_id': int(waypoint.road_id),
        'section_id': int(waypoint.section_id),
        'lane_id': int(waypoint.lane_id),
        'lane_width_m': round(float(waypoint.lane_width), 4),
        'is_junction': int(bool(waypoint.is_junction)),
        'traffic_light_state': traffic_light_state,
        'decision_state': decision.get('decision_state', ''),
        'decision_action': decision.get('decision_action', ''),
        'decision_reason': decision.get('decision_reason', ''),
        'detected_obstacle_id': decision.get('detected_obstacle_id', ''),
        'detected_obstacle_type': decision.get('detected_obstacle_type', ''),
        'detected_obstacle_class': decision.get(
            'detected_obstacle_class', ''
        ),
        'obstacle_distance_m': decision.get('obstacle_distance_m', ''),
        'obstacle_longitudinal_m': decision.get(
            'obstacle_longitudinal_m', ''
        ),
        'obstacle_lateral_m': decision.get('obstacle_lateral_m', ''),
        'time_to_collision_s': decision.get('time_to_collision_s', ''),
        'target_lane_id': decision.get('target_lane_id', ''),
        'lane_change_direction': decision.get('lane_change_direction', ''),
        'left_lane_available': decision.get('left_lane_available', 0),
        'right_lane_available': decision.get('right_lane_available', 0),
        'left_lane_safe': decision.get('left_lane_safe', 0),
        'right_lane_safe': decision.get('right_lane_safe', 0),
        'rule_violation': decision.get('rule_violation', 0),
        'violation_type': decision.get('violation_type', ''),
        'violation_reason': decision.get('violation_reason', ''),
        'blocked_duration_s': decision.get('blocked_duration_s', ''),
        'emergency_maneuver': decision.get('emergency_maneuver', ''),
        'emergency_target_speed_kmh': decision.get(
            'emergency_target_speed_kmh', ''
        ),
        'return_to_lane_success': decision.get(
            'return_to_lane_success', ''
        ),
        'scenario_id': scene.get('scenario_id', ''),
        'scenario_type': scene.get('scenario_type', ''),
        'difficulty': scene.get('difficulty', ''),
        'obstacle_seed': scene.get('seed', ''),
        'obstacle_count': len(objects),
        'pedestrian_count': pedestrian_count(scene),
        'nearest_obstacle_distance_m': nearest_obstacle_distance(scene),
        'collision_count': collision_total,
        'lane_invasion_count': invasion_total,
    }


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else int(time.time() * 1000) & 0x7fffffff
    rng = random.Random(seed)
    client, world = connect_to_ready_world(args)
    world_map = world.get_map()
    original_settings = world.get_settings()

    episode_id = 'episode_%s_%s' % (timestamp_id(), uuid.uuid4().hex[:6])
    route_id = 'route_%s' % uuid.uuid4().hex[:10]
    paths = make_paths(args, episode_id)
    # 仅供 build_row 生成相对路径，不会写入参数文件。
    args._paths = paths

    ego = None
    cameras = {}
    event_sensors = []
    obstacle_generator = None
    rows = []
    obstacle_rows = []
    event_log = []
    completion_error = None
    started_utc = utc_now()
    route_info = None
    collision_total = 0
    invasion_total = 0
    stationary_skipped = 0
    stationary_episode_count = 0
    stationary_samples_saved = 0
    label_counts = Counter()
    obstacle_scene_count = 0
    obstacle_recovery_count = 0
    obstacle_refresh_reason_counts = Counter()
    collision_queue = None
    invasion_queue = None
    saved = 0

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)
        world.set_weather(WEATHERS[args.weather])
        world.tick()

        print('正在沿道路拓扑寻找包含左转、直行、右转的路线...')
        start, end, route, route_length_m = find_complete_route(
            world_map, rng, args.route_attempts, args.min_route_length
        )
        maneuver_sequence = compressed_maneuvers(route)
        route_info = {
            'route_id': route_id,
            'map': world_map.name,
            'start_transform': transform_dict(start),
            'end_transform': transform_dict(end),
            'start_same_direction_lane_count': same_direction_lane_count(
                world_map, start
            ),
            'start_forward_distance_to_junction_m': round(
                forward_distance_to_junction(world_map, start), 3
            ),
            'planned_length_m': round(route_length_m, 3),
            'sampling_resolution_m': 2.0,
            'maneuver_sequence': maneuver_sequence,
            'required_maneuvers': list(LABEL_ORDER),
            'waypoint_count': len(route),
            'waypoints': [
                {
                    'index': index,
                    'road_option': road_option_name(option),
                    'road_id': int(waypoint.road_id),
                    'section_id': int(waypoint.section_id),
                    'lane_id': int(waypoint.lane_id),
                    'is_junction': bool(waypoint.is_junction),
                    'transform': transform_dict(waypoint.transform),
                }
                for index, (waypoint, option) in enumerate(route)
            ],
        }
        write_json(paths['route_json'], route_info)

        ego = spawn_ego(world, start, rng)
        # 同步模式中 SpawnActor 返回后，等待一个世界帧，让服务端完成 actor
        # 注册和物理状态初始化，再挂载传感器和生成相对主车的障碍物。
        world.tick()
        confirmed_ego = world.get_actor(ego.id)
        if confirmed_ego is None or not confirmed_ego.is_alive:
            raise RuntimeError(
                '主车 actor %d 生成后未能在世界帧中确认。'
                '请确认出生点未被占用并重新运行。' % ego.id
            )
        ego = confirmed_ego
        agent = RouteDrivingAgent(
            ego,
            world_map=world_map,
            target_speed=args.target_speed,
            avoidance_mode=args.avoidance_mode,
            detection_distance=args.avoidance_distance,
            pedestrian_stop_distance=args.pedestrian_stop_distance,
            pass_clearance_distance=args.pass_clearance_distance,
            lane_change_cooldown=args.lane_change_cooldown,
            emergency_rule_mode=args.emergency_rule_mode,
            emergency_wait_seconds=args.emergency_wait_seconds,
            emergency_target_speed=args.emergency_target_speed,
        )
        agent.set_global_plan(route)

        cameras, image_queues = attach_cameras(world, ego, args)
        collision_queue = queue.Queue()
        invasion_queue = queue.Queue()
        event_sensors = attach_event_sensors(
            world, ego, collision_queue, invasion_queue
        )
        world.tick()
        if not ego.is_alive:
            raise RuntimeError('挂载相机或事件传感器后主车失效')

        activation_due = None
        obstacle_respawn_due = None
        scene_started_elapsed = None
        scene_saved_start = 0

        def clear_obstacle_scene():
            nonlocal obstacle_generator, activation_due, scene_started_elapsed
            if obstacle_generator is not None:
                obstacle_generator.cleanup()
            obstacle_generator = None
            activation_due = None
            scene_started_elapsed = None

        def spawn_obstacle_scene(reason):
            nonlocal obstacle_generator, activation_due
            nonlocal obstacle_respawn_due, scene_started_elapsed
            nonlocal scene_saved_start, obstacle_scene_count
            clear_obstacle_scene()
            now = world.get_snapshot().timestamp.elapsed_seconds
            generation_context = {}
            if args.scenario == 'avoidable_traffic':
                reserved_direction = agent.find_avoidable_direction(
                    args.obstacle_distance
                )
                if reserved_direction is None:
                    obstacle_respawn_due = (
                        now + max(1.0, args.obstacle_recovery_grace)
                    )
                    obstacle_refresh_reason_counts[
                        'deferred_no_safe_lane'
                    ] += 1
                    return False
                generation_context[
                    'reserved_passing_lane_direction'
                ] = reserved_direction
            obstacle_seed = rng.randint(0, 0x7fffffff)
            obstacle_generator = ObstacleScenarioGenerator(
                world, client, seed=obstacle_seed
            )
            obstacle_generator.generate(
                ego,
                scenario_type=args.scenario,
                difficulty=args.difficulty,
                distance_m=args.obstacle_distance,
                scenario_id='%s_obstacles_%03d' % (
                    episode_id, obstacle_scene_count
                ),
                context=generation_context,
            )
            obstacle_scene_count += 1
            activation_due = now + args.obstacle_activation_delay
            obstacle_respawn_due = None
            scene_started_elapsed = now
            scene_saved_start = saved
            obstacle_refresh_reason_counts[reason] += 1
            for image_queue in image_queues.values():
                drain_queue(image_queue)
            return True

        spawn_obstacle_scene('initial')

        for _ in range(WARMUP_TICKS):
            control = agent.run_step()
            ego.apply_control(control)
            world.tick()
        for image_queue in image_queues.values():
            drain_queue(image_queue)

        ticks_per_capture = max(1, int(math.ceil(
            args.capture_interval / FIXED_DELTA_SECONDS - 1e-9
        )))
        actual_interval = ticks_per_capture * FIXED_DELTA_SECONDS
        capture_start_elapsed = world.get_snapshot().timestamp.elapsed_seconds
        candidate_index = 0
        stationary_candidates = 0
        stationary_episode_active = False
        stationary_origin = None
        stationary_started_elapsed = None
        stationary_samples_in_episode = 0

        print('=' * 72)
        print('采集开始')
        print('  主车：%s (actor %d)' % (ego.type_id, ego.id))
        print('  地图：%s；天气：%s' % (world_map.name, args.weather))
        print('  路线：%.1f 米；路口动作：%s' % (
            route_length_m, ' -> '.join(maneuver_sequence)
        ))
        print('  三方向各 %d 张；行驶拍摄间隔 %.2f 秒' % (
            args.frames, actual_interval
        ))
        print('  障碍物：%s / %s；每 %d 张或 %.1f 模拟秒刷新' % (
            args.scenario, args.difficulty, args.obstacle_refresh,
            args.obstacle_refresh_seconds
        ))
        print('  避障决策：%s（检测距离 %.1f 米）' % (
            args.avoidance_mode, args.avoidance_distance
        ))
        print('  应急违规：%s；等待 %.1f 秒；限速 %.1f km/h' % (
            args.emergency_rule_mode, args.emergency_wait_seconds,
            args.emergency_target_speed
        ))
        print('  连续停车最多保存 %d 组；障碍物堵塞 %.1f 秒后自动恢复' % (
            args.max_stationary_samples, args.obstacle_stuck_timeout
        ))
        print('  输出：%s' % paths['episode_dir'])
        print('=' * 72)

        last_control = carla.VehicleControl()
        while saved < args.frames:
            if not ego.is_alive:
                raise RuntimeError('主车在采集过程中被销毁')
            if agent.done():
                print('提示：主车已到达规划终点，将保留终点状态完成剩余采集。')

            images = None
            world_frame = None
            snapshot = None
            for _ in range(ticks_per_capture):
                current_elapsed = (
                    world.get_snapshot().timestamp.elapsed_seconds
                )
                if (obstacle_generator is None and
                        obstacle_respawn_due is not None and
                        current_elapsed >= obstacle_respawn_due):
                    spawn_obstacle_scene('stuck_recovery')
                if not agent.done():
                    last_control = agent.run_step()
                else:
                    last_control = carla.VehicleControl(
                        throttle=0.0, brake=1.0
                    )
                ego.apply_control(last_control)
                world_frame = world.tick()
                snapshot = world.get_snapshot()
                images = {
                    view: image_for_frame(image_queues[view], world_frame)
                    for view in VIEWS
                }
                if (obstacle_generator is not None and
                        activation_due is not None and
                        snapshot.timestamp.elapsed_seconds >= activation_due):
                    obstacle_generator.activate_dynamic_actors()
                    activation_due = None

            candidate_index += 1
            elapsed_since_capture = (
                snapshot.timestamp.elapsed_seconds - capture_start_elapsed
            )
            if elapsed_since_capture > args.max_simulation_seconds:
                raise RuntimeError(
                    '达到最长模拟时间 %.1f 秒，仅保存 %d/%d 组。'
                    '可增加 --max-simulation-seconds 或降低障碍物难度。' % (
                        args.max_simulation_seconds, saved, args.frames
                    )
                )

            now_elapsed = snapshot.timestamp.elapsed_seconds
            if (obstacle_generator is not None and
                    args.obstacle_refresh_seconds > 0 and
                    scene_started_elapsed is not None and
                    now_elapsed - scene_started_elapsed >=
                    args.obstacle_refresh_seconds and
                    not agent.detour_active and
                    agent.decision_snapshot().get('decision_reason') !=
                    'red_or_yellow_light'):
                scene_spawned = spawn_obstacle_scene('simulation_time')
                stationary_candidates = 0
                stationary_episode_active = False
                stationary_origin = None
                stationary_started_elapsed = None
                stationary_samples_in_episode = 0
                if scene_spawned:
                    print(
                        '  已按模拟时间刷新障碍物场景：%d' %
                        obstacle_scene_count
                    )
                continue

            current_speed_kmh = speed_mps(ego) * 3.6
            current_location = ego.get_location()
            if not stationary_episode_active:
                if current_speed_kmh < args.stationary_speed:
                    stationary_episode_active = True
                    stationary_origin = carla.Location(
                        x=current_location.x,
                        y=current_location.y,
                        z=current_location.z,
                    )
                    stationary_started_elapsed = now_elapsed
                    stationary_candidates = 0
                    stationary_samples_in_episode = 0
                    stationary_episode_count += 1
            else:
                moved_distance = current_location.distance(stationary_origin)
                if (current_speed_kmh >= args.stationary_speed and
                        moved_distance >= args.stationary_reset_distance):
                    stationary_episode_active = False
                    stationary_origin = None
                    stationary_started_elapsed = None
                    stationary_candidates = 0
                    stationary_samples_in_episode = 0

            if stationary_episode_active:
                stationary_candidates += 1
                decision = agent.decision_snapshot()
                stop_duration = now_elapsed - stationary_started_elapsed
                obstacle_blocked = (
                    decision.get('decision_reason') in
                    OBSTACLE_BLOCKING_REASONS
                )
                if (obstacle_blocked and
                        obstacle_generator is not None and
                        args.obstacle_stuck_timeout > 0 and
                        stop_duration >= args.obstacle_stuck_timeout):
                    blocking_reason = decision.get('decision_reason')
                    clear_obstacle_scene()
                    obstacle_respawn_due = (
                        now_elapsed + args.obstacle_recovery_grace
                    )
                    obstacle_recovery_count += 1
                    obstacle_refresh_reason_counts['stuck_cleanup'] += 1
                    stationary_skipped += 1
                    stationary_episode_active = False
                    stationary_origin = None
                    stationary_started_elapsed = None
                    stationary_candidates = 0
                    stationary_samples_in_episode = 0
                    print(
                        '  障碍物堵塞 %.1f 秒（%s），已清场；%.1f 秒后重新生成。' % (
                            stop_duration, blocking_reason,
                            args.obstacle_recovery_grace
                        )
                    )
                    continue
                if (stationary_samples_in_episode >=
                        args.max_stationary_samples):
                    stationary_skipped += 1
                    continue
                if ((stationary_candidates - 1) %
                        args.stationary_sample_every != 0):
                    stationary_skipped += 1
                    continue

            timestamp_ns = int(round(
                snapshot.timestamp.elapsed_seconds * 1_000_000_000
            ))
            collision_total += drain_events(
                collision_queue, event_log, timestamp_ns
            )
            invasion_total += drain_events(
                invasion_queue, event_log, timestamp_ns
            )

            label, option = future_label(
                agent, args.label_horizon_ms, speed_mps(ego)
            )
            label_counts[label] += 1
            sample_id = '%s_%08d' % (episode_id, int(world_frame))
            image_paths = {}
            for view in VIEWS:
                filename = '%s_%04d_%08d.png' % (
                    view.lower(), saved, int(world_frame)
                )
                absolute_path = os.path.join(paths['rgb_dirs'][view], filename)
                images[view].save_to_disk(absolute_path)
                image_paths[view] = relative_path(
                    absolute_path, paths['dataset_root']
                )

            scene_snapshot = (
                obstacle_generator.snapshot()
                if obstacle_generator is not None else None
            )
            waypoint = world_map.get_waypoint(
                ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if waypoint is None:
                raise RuntimeError('无法把主车当前位置投影到 Driving 车道')

            row = build_row(
                ego, last_control, waypoint, sample_id, episode_id, route_id,
                world_frame, timestamp_ns, snapshot.timestamp.elapsed_seconds,
                image_paths, label, option, args, world_map, scene_snapshot,
                agent.decision_snapshot(), collision_total, invasion_total
            )
            rows.append(row)
            obstacle_rows.append({
                'sample_id': sample_id,
                'episode_id': episode_id,
                'route_id': route_id,
                'frame_id': int(world_frame),
                'timestamp_ns': timestamp_ns,
                # snapshot() 会复用并更新同一个 scene 字典，必须深拷贝，
                # 否则之前帧会被后续位置覆盖。
                'scenario': copy.deepcopy(scene_snapshot),
            })
            saved += 1
            if stationary_episode_active:
                stationary_samples_in_episode += 1
                stationary_samples_saved += 1

            if (obstacle_generator is not None and
                    args.obstacle_refresh > 0 and saved < args.frames and
                    saved - scene_saved_start >= args.obstacle_refresh and
                    not agent.detour_active):
                if spawn_obstacle_scene('saved_frames'):
                    print('  已刷新障碍物场景：%d' % obstacle_scene_count)

            if saved % 10 == 0 or saved == args.frames:
                print(
                    '  %3d/%d  frame=%d  speed=%.1f km/h  label=%s  '
                    'obstacles=%d  decision=%s' % (
                        saved, args.frames, world_frame, current_speed_kmh,
                        label, row['obstacle_count'], row['decision_action']
                    )
                )

    except Exception as error:
        completion_error = str(error)
        raise
    finally:
        finished_utc = utc_now()
        final_timestamp_ns = int(time.time() * 1_000_000_000)
        if collision_queue is not None:
            collision_total += drain_events(
                collision_queue, event_log, final_timestamp_ns
            )
        if invasion_queue is not None:
            invasion_total += drain_events(
                invasion_queue, event_log, final_timestamp_ns
            )
        try:
            if obstacle_generator is not None:
                obstacle_generator.cleanup()
        except RuntimeError:
            pass
        stop_and_destroy(client, list(cameras.values()) + event_sensors)
        if ego is not None:
            stop_and_destroy(client, [ego])
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

        if paths and route_info is not None:
            try:
                with open(paths['control_csv'], 'w', newline='',
                          encoding='utf-8-sig') as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                if rows:
                    append_manifest(paths['manifest_csv'], rows)
                write_jsonl(paths['obstacles_jsonl'], obstacle_rows)
                write_jsonl(paths['events_jsonl'], event_log)
                metadata = {
                    'schema_version': SCHEMA_VERSION,
                    'dataset_version': DATASET_VERSION,
                    'episode_id': episode_id,
                    'route_id': route_id,
                    'split': args.split,
                    'seed': seed,
                    'started_at_utc': started_utc,
                    'finished_at_utc': finished_utc,
                    'complete': completion_error is None and len(rows) == args.frames,
                    'error': completion_error,
                    'map': world_map.name,
                    'weather_name': args.weather,
                    'weather_parameters': weather_dict(WEATHERS[args.weather]),
                    'label_order': list(LABEL_ORDER),
                    'label_horizon_ms': args.label_horizon_ms,
                    'field_units': {
                        'timestamp_ns': 'simulation nanoseconds',
                        'speed': 'km/h',
                        'speed_mps': 'm/s',
                        'speed_kmh': 'km/h',
                        'steer': 'CARLA normalized [-1, 1]',
                        'throttle': 'CARLA normalized [0, 1]',
                        'brake': 'CARLA normalized [0, 1]',
                    },
                    'model_field_status': {
                        'event_path': 'empty; no DVS sensor in this RGB capture',
                        'snn_label': 'missing until offline SNN inference',
                        'available_mask': 0,
                        'ann_probabilities': 'empty until offline ANN inference',
                    },
                    'camera_views': list(VIEWS),
                    'camera_transforms': {
                        view: transform_dict(CAMERA_TRANSFORMS[view])
                        for view in VIEWS
                    },
                    'image_width': args.width,
                    'image_height': args.height,
                    'fov_deg': args.fov,
                    'requested_frames_per_view': args.frames,
                    'saved_frames_per_view': len(rows),
                    'saved_image_total': len(rows) * len(VIEWS),
                    'capture_interval_seconds_moving': args.capture_interval,
                    'stationary_speed_threshold_kmh': args.stationary_speed,
                    'stationary_sample_every': args.stationary_sample_every,
                    'max_stationary_samples_per_stop': (
                        args.max_stationary_samples
                    ),
                    'stationary_reset_distance_m': (
                        args.stationary_reset_distance
                    ),
                    'stationary_episode_count': stationary_episode_count,
                    'stationary_samples_saved': stationary_samples_saved,
                    'stationary_candidates_skipped': stationary_skipped,
                    'target_speed_kmh': args.target_speed,
                    'avoidance_policy': {
                        'version': RouteDrivingAgent.POLICY_VERSION,
                        'mode': args.avoidance_mode,
                        'detection_distance_m': args.avoidance_distance,
                        'pedestrian_stop_distance_m': (
                            args.pedestrian_stop_distance
                        ),
                        'pass_clearance_distance_m': (
                            args.pass_clearance_distance
                        ),
                        'lane_change_cooldown_s': (
                            args.lane_change_cooldown
                        ),
                        'obstacle_stuck_timeout_s': (
                            args.obstacle_stuck_timeout
                        ),
                        'obstacle_recovery_grace_s': (
                            args.obstacle_recovery_grace
                        ),
                        'obstacle_recovery_count': obstacle_recovery_count,
                        'lane_clearance_method': (
                            'curved_waypoint_corridor_with_rotated_bbox'
                        ),
                        'emergency_rule_mode': args.emergency_rule_mode,
                        'emergency_wait_seconds': (
                            args.emergency_wait_seconds
                        ),
                        'emergency_target_speed_kmh': (
                            args.emergency_target_speed
                        ),
                        'pedestrian_rule': (
                            'yield until crossing pedestrian clears; '
                            'never proactively bypass'
                        ),
                        'lane_change_rule': (
                            'prevalidated complete detour on same-direction '
                            'Driving lane, legal markings, safe front/rear '
                            'gap, outside junction'
                        ),
                        'emergency_rule': (
                            'after timeout, cross solid marking only into '
                            'a safe same-direction Driving lane; never for '
                            'pedestrians, traffic lights, or junctions'
                        ),
                    },
                    'decision_action_counts': dict(Counter(
                        row['decision_action'] for row in rows
                    )),
                    'decision_state_counts': dict(Counter(
                        row['decision_state'] for row in rows
                    )),
                    'rule_violation_sample_count': sum(
                        int(row.get('rule_violation', 0)) for row in rows
                    ),
                    'violation_type_counts': dict(Counter(
                        row['violation_type'] for row in rows
                        if row.get('violation_type')
                    )),
                    'label_counts': dict(label_counts),
                    'route_summary': {
                        key: value for key, value in (route_info or {}).items()
                        if key != 'waypoints'
                    },
                    'obstacle_config': {
                        'scenario': args.scenario,
                        'difficulty': args.difficulty,
                        'distance_m': args.obstacle_distance,
                        'refresh_every_saved_frames': args.obstacle_refresh,
                        'refresh_every_simulation_seconds': (
                            args.obstacle_refresh_seconds
                        ),
                        'activation_delay_seconds': args.obstacle_activation_delay,
                        'scene_count': obstacle_scene_count,
                        'refresh_reason_counts': dict(
                            obstacle_refresh_reason_counts
                        ),
                    },
                    'collision_event_count': collision_total,
                    'lane_invasion_event_count': invasion_total,
                    'files': {
                        'control_csv': relative_path(
                            paths['control_csv'], paths['dataset_root']
                        ),
                        'manifest_csv': relative_path(
                            paths['manifest_csv'], paths['dataset_root']
                        ),
                        'route_json': relative_path(
                            paths['route_json'], paths['dataset_root']
                        ),
                        'obstacles_jsonl': relative_path(
                            paths['obstacles_jsonl'], paths['dataset_root']
                        ),
                        'events_jsonl': relative_path(
                            paths['events_jsonl'], paths['dataset_root']
                        ),
                    },
                }
                write_json(paths['metadata_json'], metadata)
            except Exception as save_error:
                print('警告：保存最终元数据失败：%s' % save_error)

    print('=' * 72)
    print('采集完成：每个方向 %d 张，共 %d 张图片' % (
        len(rows), len(rows) * len(VIEWS)
    ))
    print('标签分布：%s' % dict(label_counts))
    print('障碍物场景：%d 轮；碰撞：%d；越线：%d' % (
        obstacle_scene_count, collision_total, invasion_total
    ))
    print('输出：%s' % paths['episode_dir'])
    print('=' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
