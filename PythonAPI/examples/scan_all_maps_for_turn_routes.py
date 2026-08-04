#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
依次加载 CARLA 的全部地图，寻找同时覆盖左转、直行、右转的路线。

CMD 示例：
    C:\Users\yangz\anaconda3\envs\carla-test\python.exe ^
      D:\Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\scan_all_maps_for_turn_routes.py
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from capture_route_with_obstacles import (
    LABEL_ORDER,
    carla,
    compressed_maneuvers,
    find_complete_route,
    normalized_yaw_delta,
    road_option_name,
    transform_dict,
)


SCHEMA_VERSION = 'carla_all_map_turn_route_scan_v1.0'


def parse_args():
    parser = argparse.ArgumentParser(
        description='扫描 CARLA 地图，寻找包含 left/straight/right 的路线。'
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument(
        '--maps', default='all',
        help='all 或逗号分隔的地图名，例如 Town03,Town05,Town10HD_Opt'
    )
    parser.add_argument(
        '--exclude-opt', action='store_true',
        help='排除名称以 _Opt 结尾的优化版地图，避免与普通版重复'
    )
    parser.add_argument('--attempts', type=int, default=150,
                        help='每张地图最多尝试的路线数（默认 150）')
    parser.add_argument('--min-route-length', type=float, default=300.0,
                        help='合格路线最短长度，单位米（默认 300）')
    parser.add_argument('--seed', type=int, default=20260803)
    parser.add_argument('--list-only', action='store_true',
                        help='只列出服务器可用地图，不进行扫描')
    parser.add_argument('--open-best', action='store_true',
                        help='扫描结束后加载成功路线最长的地图')
    parser.add_argument('--output', default=None,
                        help='结果 JSON 路径；默认写入 examples/output/map_route_scan')
    args = parser.parse_args()
    if args.attempts <= 0:
        parser.error('--attempts 必须大于 0')
    if args.min_route_length <= 0:
        parser.error('--min-route-length 必须大于 0')
    return args


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def short_map_name(map_path):
    return map_path.replace('\\', '/').rstrip('/').split('/')[-1]


def select_maps(available_maps, requested, exclude_opt):
    available = sorted(set(available_maps), key=lambda item: short_map_name(item).lower())
    if exclude_opt:
        available = [
            item for item in available
            if not short_map_name(item).lower().endswith('_opt')
        ]
    if requested.strip().lower() == 'all':
        return available

    wanted = [item.strip() for item in requested.split(',') if item.strip()]
    selected = []
    missing = []
    for name in wanted:
        match = next((
            path for path in available
            if short_map_name(path).lower() == name.lower()
            or path.lower() == name.lower()
        ), None)
        if match is None:
            missing.append(name)
        elif match not in selected:
            selected.append(match)
    if missing:
        raise RuntimeError(
            '服务器中没有这些地图：%s' % ', '.join(missing)
        )
    return selected


def route_turn_entries(route):
    entries = []
    previous = None
    for index, (waypoint, option) in enumerate(route):
        name = road_option_name(option).lower()
        if name not in LABEL_ORDER or name == previous:
            continue
        entries.append({
            'route_index': index,
            'maneuver': name,
            'road_id': int(waypoint.road_id),
            'section_id': int(waypoint.section_id),
            'lane_id': int(waypoint.lane_id),
            'is_junction': bool(waypoint.is_junction),
            'transform': transform_dict(waypoint.transform),
        })
        previous = name
    return entries


def junction_maneuver_inventory(world_map):
    """独立于路线搜索，枚举地图所有 Junction 实际支持的动作。"""
    junctions = {}
    for waypoint in world_map.generate_waypoints(5.0):
        if not waypoint.is_junction:
            continue
        junction = waypoint.get_junction()
        if junction is None:
            continue
        junction_id = int(junction.id)
        if junction_id in junctions:
            continue
        maneuvers = []
        connectors = []
        seen_connectors = set()
        for entry, exit_waypoint in junction.get_waypoints(
                carla.LaneType.Driving):
            key = (
                int(entry.road_id), int(entry.lane_id),
                int(exit_waypoint.road_id), int(exit_waypoint.lane_id)
            )
            if key in seen_connectors:
                continue
            seen_connectors.add(key)
            delta = normalized_yaw_delta(
                entry.transform.rotation.yaw,
                exit_waypoint.transform.rotation.yaw
            )
            if abs(delta) <= 35.0:
                maneuver = 'straight'
            elif delta > 0.0:
                maneuver = 'left'
            else:
                maneuver = 'right'
            maneuvers.append(maneuver)
            connectors.append({
                'maneuver': maneuver,
                'heading_change_deg': round(float(delta), 3),
                'entry_road_id': int(entry.road_id),
                'entry_lane_id': int(entry.lane_id),
                'exit_road_id': int(exit_waypoint.road_id),
                'exit_lane_id': int(exit_waypoint.lane_id),
                'entry_transform': transform_dict(entry.transform),
                'exit_transform': transform_dict(exit_waypoint.transform),
            })
        junctions[junction_id] = {
            'junction_id': junction_id,
            'maneuvers': sorted(set(maneuvers)),
            'connector_count': len(connectors),
            'connectors': connectors,
        }

    available = sorted(set(
        maneuver
        for junction in junctions.values()
        for maneuver in junction['maneuvers']
    ))
    counts = {
        label: sum(
            1
            for junction in junctions.values()
            for connector in junction['connectors']
            if connector['maneuver'] == label
        )
        for label in LABEL_ORDER
    }
    return {
        'junction_count': len(junctions),
        'available_maneuvers': available,
        'supports_left_straight_right': set(LABEL_ORDER).issubset(available),
        'connector_counts': counts,
        'junctions': list(junctions.values()),
    }


def default_output_path():
    output_dir = os.path.join(SCRIPT_DIR, 'output', 'map_route_scan')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        'all_maps_%s.json' % datetime.now().strftime('%Y%m%d_%H%M%S')
    )


def save_result(path, result):
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(absolute, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return absolute


def main():
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(90.0)

    available_maps = list(client.get_available_maps())
    selected_maps = select_maps(
        available_maps, args.maps, args.exclude_opt
    )
    if not selected_maps:
        raise RuntimeError('没有可扫描的地图')

    print('CARLA 服务器可用地图：%d 张' % len(available_maps))
    for map_path in selected_maps:
        print('  %s' % short_map_name(map_path))
    if args.list_only:
        return 0

    result = {
        'schema_version': SCHEMA_VERSION,
        'generated_at_utc': utc_now(),
        'host': args.host,
        'port': args.port,
        'seed': args.seed,
        'attempts_per_map': args.attempts,
        'minimum_route_length_m': args.min_route_length,
        'required_maneuvers': list(LABEL_ORDER),
        'available_maps': available_maps,
        'scanned_maps': [short_map_name(item) for item in selected_maps],
        'results': [],
    }

    print('\n开始逐张加载和扫描。切换地图时 CARLA 窗口短暂无响应是正常现象。')
    for index, map_path in enumerate(selected_maps, 1):
        name = short_map_name(map_path)
        started = time.time()
        print('\n[%d/%d] %s' % (index, len(selected_maps), name))
        item = {
            'map_path': map_path,
            'map_name': name,
            'success': False,
        }
        try:
            world = client.load_world(map_path)
            world_map = world.get_map()
            inventory = junction_maneuver_inventory(world_map)
            item['junction_inventory'] = inventory
            print(
                '  路口能力：%d 个 Junction；left=%d, straight=%d, right=%d' % (
                    inventory['junction_count'],
                    inventory['connector_counts']['left'],
                    inventory['connector_counts']['straight'],
                    inventory['connector_counts']['right'],
                )
            )
            map_seed = args.seed + index * 100003
            start, end, route, length_m = find_complete_route(
                world_map,
                random.Random(map_seed),
                args.attempts,
                args.min_route_length,
            )
            maneuvers = compressed_maneuvers(route)
            required_found = set(LABEL_ORDER).issubset(set(maneuvers))
            meets_length = length_m >= args.min_route_length
            item.update({
                'success': bool(required_found and meets_length),
                'map_supports_all_maneuvers': inventory[
                    'supports_left_straight_right'
                ],
                'continuous_route_found': bool(required_found),
                'meets_minimum_length': bool(meets_length),
                'map_seed': map_seed,
                'planned_length_m': round(float(length_m), 3),
                'waypoint_count': len(route),
                'maneuver_sequence': maneuvers,
                'start_transform': transform_dict(start),
                'end_transform': transform_dict(end),
                'turn_entries': route_turn_entries(route),
            })
            if required_found and meets_length:
                print('  成功：%.1f 米，动作序列：%s' % (
                    length_m, ' -> '.join(maneuvers)
                ))
            else:
                print('  找到三类动作，但路线长度不足：%.1f/%g 米' % (
                    length_m, args.min_route_length
                ))
        except Exception as error:
            if 'junction_inventory' in item:
                item['map_supports_all_maneuvers'] = item[
                    'junction_inventory'
                ]['supports_left_straight_right']
                item['continuous_route_found'] = False
            item['error'] = str(error)
            print('  未找到：%s' % error)
        item['scan_seconds'] = round(time.time() - started, 3)
        result['results'].append(item)

        # 每扫描完一张地图就保存一次，扫描中断时也能保留已有结果。
        output_path = args.output or default_output_path()
        # default_output_path 只应在第一次确定；后续复用同一路径。
        args.output = output_path
        save_result(output_path, result)

    successes = [item for item in result['results'] if item['success']]
    successes.sort(
        key=lambda item: item.get('planned_length_m', 0.0), reverse=True
    )
    result['successful_map_count'] = len(successes)
    result['failed_map_count'] = len(result['results']) - len(successes)
    result['recommended_map'] = successes[0] if successes else None
    output_path = save_result(args.output or default_output_path(), result)

    print('\n' + '=' * 72)
    print('扫描完成：%d/%d 张地图找到合格路线' % (
        len(successes), len(result['results'])
    ))
    if successes:
        print('推荐地图：%s（%.1f 米）' % (
            successes[0]['map_name'], successes[0]['planned_length_m']
        ))
    print('结果：%s' % output_path)
    print('=' * 72)

    if args.open_best and successes:
        print('正在加载推荐地图：%s' % successes[0]['map_name'])
        client.load_world(successes[0]['map_path'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
