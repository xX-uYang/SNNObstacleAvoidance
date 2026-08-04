#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
基于 capture_multi_weather_v3 的稳定采集版。

默认采集内容：
    6 种天气 x 3 个拍摄方向 x 每方向 100 张 = 1800 张 PNG

特点：
    1. 保留 v3 的三相机位置、天气组合和简洁目录结构。
    2. 使用 CARLA 同步模式，让三个方向保存同一个世界帧。
    3. 只在主车指定半径内随机生成行人，并可随车辆位置定期刷新。
    4. 拍照间隔可调，默认每 1.0 秒保存一组三方向同步照片。
    5. 检测到车辆静止时自动降低保存频率，减少红灯重复照片。
    6. 每种天气结束后严格检查三个方向是否各有 100 张。
    7. 只清理由本脚本生成的 actor，不会删除场景中已有 actor。

推荐运行方式（使用已安装 CARLA 模块的 Conda 环境）：
    C:\Users\yangz\anaconda3\envs\carla-test\python.exe capture_multi_weather_v4.py
"""

import argparse
import glob
import math
import os
import queue
import random
import sys
import time
from datetime import datetime


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


IMG_WIDTH = 1280
IMG_HEIGHT = 720
FRAMES_PER_WEATHER = 100
WALKERS_PER_WEATHER = 12
FIXED_DELTA_SECONDS = 0.1
CAPTURE_INTERVAL_SECONDS = 1.0
SENSOR_TIMEOUT_SECONDS = 5.0
WARMUP_TICKS = 10
WALKER_RADIUS_METERS = 50.0
WALKER_MIN_DISTANCE_METERS = 5.0
WALKER_REFRESH_FRAMES = 10
STATIONARY_SPEED_KMH = 1.0
STATIONARY_SAMPLE_EVERY = 5

VIEWS = ('STRAIGHT', 'LEFT', 'RIGHT')

WEATHERS = [
    ('ClearNoon', carla.WeatherParameters.ClearNoon),
    ('CloudyNoon', carla.WeatherParameters.CloudyNoon),
    ('WetSunset', carla.WeatherParameters.WetSunset),
    ('ClearSunset', carla.WeatherParameters.ClearSunset),
    ('SoftRainSunset', carla.WeatherParameters.SoftRainSunset),
    ('HardRainNoon', carla.WeatherParameters.HardRainNoon),
]

CAMERA_TRANSFORMS = {
    'STRAIGHT': carla.Transform(
        carla.Location(x=1.5, z=1.2),
        carla.Rotation()
    ),
    'LEFT': carla.Transform(
        carla.Location(y=-1.0, z=1.2),
        carla.Rotation(yaw=-90.0)
    ),
    'RIGHT': carla.Transform(
        carla.Location(y=1.0, z=1.4),
        carla.Rotation(pitch=-5.0, yaw=85.0)
    ),
}

BAD_VEHICLE_WORDS = (
    'bike', 'motorcycle', 'micro', 'gazelle', 'harley', 'crossbike',
    'diamondback', 'low_rider', 'omafiets', 'carlamotors'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='采集 6 种天气、3 个方向的 CARLA RGB 图片，并随机生成行人。'
    )
    parser.add_argument('--host', default='127.0.0.1', help='CARLA 主机地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA RPC 端口')
    parser.add_argument('--tm-port', type=int, default=8000, help='Traffic Manager 端口')
    parser.add_argument(
        '--frames', type=int, default=FRAMES_PER_WEATHER,
        help='每种天气、每个方向保存的图片数（默认 100）'
    )
    parser.add_argument(
        '--walkers', type=int, default=WALKERS_PER_WEATHER,
        help='每种天气随机生成的行人数（默认 12）'
    )
    parser.add_argument(
        '--walker-radius', type=float, default=WALKER_RADIUS_METERS,
        help='只在主车周围此半径内生成行人，单位米（默认 50）'
    )
    parser.add_argument(
        '--walker-min-distance', type=float,
        default=WALKER_MIN_DISTANCE_METERS,
        help='行人与主车的最小出生距离，单位米（默认 5）'
    )
    parser.add_argument(
        '--walker-refresh', type=int, default=WALKER_REFRESH_FRAMES,
        help='每保存多少张后在主车周围刷新行人；0 表示不刷新（默认 10）'
    )
    parser.add_argument(
        '--capture-interval', type=float,
        default=CAPTURE_INTERVAL_SECONDS,
        help='相邻照片的模拟时间间隔，单位秒（默认 1.0）'
    )
    parser.add_argument(
        '--stationary-speed', type=float, default=STATIONARY_SPEED_KMH,
        help='低于此车速视为静止，单位 km/h（默认 1.0）'
    )
    parser.add_argument(
        '--stationary-sample-every', type=int,
        default=STATIONARY_SAMPLE_EVERY,
        help='静止时每几个拍摄时刻保存一组；1 表示不降采样（默认 5）'
    )
    parser.add_argument('--seed', type=int, default=None, help='固定随机种子，便于复现')
    parser.add_argument(
        '--output', default=None,
        help='输出目录；默认在 examples/output 下新建带时间戳的目录'
    )
    args = parser.parse_args()

    if args.frames <= 0:
        parser.error('--frames 必须大于 0')
    if args.walkers < 0:
        parser.error('--walkers 不能小于 0')
    if args.walker_radius <= 0:
        parser.error('--walker-radius 必须大于 0')
    if args.walker_min_distance < 0:
        parser.error('--walker-min-distance 不能小于 0')
    if args.walker_min_distance >= args.walker_radius:
        parser.error('--walker-min-distance 必须小于 --walker-radius')
    if args.walker_refresh < 0:
        parser.error('--walker-refresh 不能小于 0')
    if args.capture_interval < FIXED_DELTA_SECONDS:
        parser.error(
            '--capture-interval 不能小于 %.1f 秒' % FIXED_DELTA_SECONDS
        )
    if args.stationary_speed < 0:
        parser.error('--stationary-speed 不能小于 0')
    if args.stationary_sample_every <= 0:
        parser.error('--stationary-sample-every 必须大于 0')
    return args


def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def make_output_root(path):
    if path:
        output_root = os.path.abspath(path)
    else:
        output_root = os.path.join(
            SCRIPT_DIR, 'output', 'multi_weather_%s' % timestamp()
        )
    os.makedirs(output_root, exist_ok=True)
    return output_root


def destroy_actors(client, actors):
    """停止控制器/传感器，并批量销毁本脚本创建的 actor。"""
    alive = []
    for actor in actors:
        if actor is None:
            continue
        try:
            if actor.type_id.startswith('sensor.'):
                actor.stop()
            elif actor.type_id.startswith('controller.ai.walker'):
                actor.stop()
        except RuntimeError:
            pass
        try:
            if actor.is_alive:
                alive.append(actor)
        except RuntimeError:
            pass

    if alive:
        client.apply_batch_sync(
            [carla.command.DestroyActor(actor.id) for actor in alive],
            True
        )


def vehicle_blueprints(world):
    blueprints = []
    for blueprint in world.get_blueprint_library().filter('vehicle.*'):
        if any(word in blueprint.id for word in BAD_VEHICLE_WORDS):
            continue
        if blueprint.has_attribute('number_of_wheels'):
            try:
                # ActorAttribute 的字符串形式包含描述信息，必须直接转 int。
                if int(blueprint.get_attribute('number_of_wheels')) != 4:
                    continue
            except (TypeError, ValueError):
                continue
        blueprints.append(blueprint)
    return blueprints


def spawn_ego_vehicle(world):
    blueprints = vehicle_blueprints(world)
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)

    if not blueprints:
        raise RuntimeError('没有找到可用的四轮车辆 blueprint')
    if not spawn_points:
        raise RuntimeError('当前地图没有车辆出生点')

    for spawn_point in spawn_points:
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', 'hero')
        ego = world.try_spawn_actor(blueprint, spawn_point)
        if ego is not None:
            return ego
    raise RuntimeError('所有车辆出生点均被占用，主车生成失败')


def random_navigation_transforms(world, center, count, min_distance, radius):
    """只返回主车指定距离范围内的随机导航网格位置。"""
    nearby = []
    max_attempts = max(1000, count * 200)

    for _ in range(max_attempts):
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        distance = location.distance(center)
        if distance < min_distance or distance > radius:
            continue
        if any(
                location.distance(item.location) < 1.5
                for item in nearby):
            continue
        nearby.append(carla.Transform(location))
        if len(nearby) >= count:
            break
    return nearby


def spawn_walkers(world, client, count, center, min_distance, radius):
    """按照 CARLA generate_traffic.py 的方式随机生成行人和 AI 控制器。"""
    if count == 0:
        return []

    walker_blueprints = list(
        world.get_blueprint_library().filter('walker.pedestrian.*')
    )
    if not walker_blueprints:
        print('    警告：当前地图没有行人 blueprint')
        return []

    spawn_transforms = random_navigation_transforms(
        world, center, count, min_distance, radius
    )
    if len(spawn_transforms) < count:
        print(
            '    提示：%g-%g 米范围内只找到 %d/%d 个可用行人点' % (
                min_distance, radius, len(spawn_transforms), count
            )
        )
    walker_batch = []
    requested_speeds = []

    for transform in spawn_transforms:
        blueprint = random.choice(walker_blueprints)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')

        speed = 1.4
        if blueprint.has_attribute('speed'):
            values = blueprint.get_attribute('speed').recommended_values
            if len(values) > 2 and random.random() < 0.15:
                speed = float(values[2])
            elif len(values) > 1:
                speed = float(values[1])
        requested_speeds.append(speed)
        walker_batch.append(carla.command.SpawnActor(blueprint, transform))

    if not walker_batch:
        return []

    walker_results = client.apply_batch_sync(walker_batch, True)
    walkers = []
    for result, speed in zip(walker_results, requested_speeds):
        if not result.error:
            walkers.append((result.actor_id, speed))

    if not walkers:
        print('    警告：随机行人全部生成失败')
        return []

    controller_blueprint = world.get_blueprint_library().find(
        'controller.ai.walker'
    )
    controller_batch = [
        carla.command.SpawnActor(
            controller_blueprint, carla.Transform(), walker_id
        )
        for walker_id, _ in walkers
    ]
    controller_results = client.apply_batch_sync(controller_batch, True)

    # 与官方 generate_traffic.py 一致：等待客户端收到新行人的最终变换。
    world.tick()

    managed_actors = []
    orphan_walker_ids = []
    for walker_info, result in zip(walkers, controller_results):
        walker_id, speed = walker_info
        if result.error:
            orphan_walker_ids.append(walker_id)
            continue

        walker = world.get_actor(walker_id)
        controller = world.get_actor(result.actor_id)
        if walker is None or controller is None:
            orphan_walker_ids.append(walker_id)
            orphan_walker_ids.append(result.actor_id)
            continue

        controller.start()
        target = world.get_random_location_from_navigation()
        if target is not None:
            controller.go_to_location(target)
        controller.set_max_speed(speed)
        managed_actors.extend([controller, walker])

    if orphan_walker_ids:
        client.apply_batch_sync(
            [carla.command.DestroyActor(actor_id) for actor_id in orphan_walker_ids],
            True
        )

    print(
        '    随机行人：%d/%d 个（距主车 %g-%g 米）' % (
            len(managed_actors) // 2, count, min_distance, radius
        )
    )
    return managed_actors


def attach_cameras(world, ego):
    blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', str(IMG_WIDTH))
    blueprint.set_attribute('image_size_y', str(IMG_HEIGHT))
    blueprint.set_attribute('fov', '90')
    blueprint.set_attribute('sensor_tick', '0.0')

    cameras = {}
    image_queues = {}
    try:
        for view in VIEWS:
            camera = world.spawn_actor(
                blueprint,
                CAMERA_TRANSFORMS[view],
                attach_to=ego
            )
            image_queue = queue.Queue()
            camera.listen(lambda image, q=image_queue: q.put(image))
            cameras[view] = camera
            image_queues[view] = image_queue
    except Exception:
        for camera in cameras.values():
            try:
                camera.stop()
                camera.destroy()
            except RuntimeError:
                pass
        raise
    return cameras, image_queues


def drain_queue(image_queue):
    while True:
        try:
            image_queue.get_nowait()
        except queue.Empty:
            return


def image_for_frame(image_queue, expected_frame):
    deadline = time.time() + SENSOR_TIMEOUT_SECONDS
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise RuntimeError(
                '等待相机世界帧 %d 超时' % expected_frame
            )
        try:
            image = image_queue.get(timeout=remaining)
        except queue.Empty:
            raise RuntimeError(
                '等待相机世界帧 %d 超时' % expected_frame
            )

        if image.frame < expected_frame:
            continue
        if image.frame > expected_frame:
            raise RuntimeError(
                '相机跳过世界帧 %d，收到 %d' % (expected_frame, image.frame)
            )
        return image


def create_run_paths(output_root, run_name):
    paths = {}
    for view in VIEWS:
        path = os.path.join(output_root, run_name, view)
        os.makedirs(path, exist_ok=True)
        paths[view] = path
    return paths


def vehicle_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return 3.6 * (
        velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
    ) ** 0.5


def capture_frames(world, client, ego, image_queues, paths, run_name,
                   frame_count, args, owned_actors, walker_actors):
    for _ in range(WARMUP_TICKS):
        world.tick()
    for image_queue in image_queues.values():
        drain_queue(image_queue)

    ticks_per_capture = max(
        1,
        int(math.ceil(
            args.capture_interval / FIXED_DELTA_SECONDS - 1e-9
        ))
    )
    actual_interval = ticks_per_capture * FIXED_DELTA_SECONDS
    print(
        '    开始采集：每个方向 %d 张，帧间隔 %.1f 秒...' % (
            frame_count, actual_interval
        )
    )
    saved = 0
    stationary_candidates = 0
    total_stationary_skipped = 0
    while saved < frame_count:
        if not ego.is_alive:
            raise RuntimeError('主车在采集过程中被销毁')

        # 每个 tick 都取走相机帧，只保存间隔结束时的最后一组。
        # 这样既能拉开照片距离，也不会让三个相机的队列持续堆积。
        images = None
        world_frame = None
        for _ in range(ticks_per_capture):
            world_frame = world.tick()
            images = {}
            for view in VIEWS:
                images[view] = image_for_frame(
                    image_queues[view], world_frame
                )

        speed = vehicle_speed_kmh(ego)
        if speed < args.stationary_speed:
            stationary_candidates += 1
            save_stationary_frame = (
                (stationary_candidates - 1) %
                args.stationary_sample_every == 0
            )
            if stationary_candidates == 1:
                print(
                    '      检测到车辆静止（%.1f km/h），静止画面按 1/%d 保存' % (
                        speed, args.stationary_sample_every
                    )
                )
            if not save_stationary_frame:
                total_stationary_skipped += 1
                continue
        else:
            if stationary_candidates > 1:
                print(
                    '      车辆恢复行驶（%.1f km/h），恢复正常保存频率' % speed
                )
            stationary_candidates = 0

        for view in VIEWS:
            filename = '%s_%04d_frame%08d.png' % (
                run_name, saved, world_frame
            )
            images[view].save_to_disk(os.path.join(paths[view], filename))
        saved += 1

        if (args.walker_refresh > 0 and saved < frame_count and
                saved % args.walker_refresh == 0):
            destroy_actors(client, walker_actors)
            walker_actors = spawn_walkers(
                world,
                client,
                args.walkers,
                ego.get_location(),
                args.walker_min_distance,
                args.walker_radius
            )
            owned_actors.extend(walker_actors)
            for image_queue in image_queues.values():
                drain_queue(image_queue)
            print('      已在主车当前位置周围刷新行人')

        if saved % 25 == 0 or saved == frame_count:
            print(
                '      %d/%d，世界帧 %d，车速 %.0f km/h' % (
                    saved, frame_count, world_frame, speed
                )
            )
    if total_stationary_skipped:
        print(
            '    静止降采样：跳过 %d 组重复画面，最终仍保存 %d 组' % (
                total_stationary_skipped, saved
            )
        )
    return saved


def png_count(path):
    return len([
        name for name in os.listdir(path)
        if name.lower().endswith('.png')
    ])


def run_one_weather(world, client, traffic_manager, weather_index,
                    weather_name, weather, output_root, args):
    run_name = 'run%02d_%s' % (weather_index, weather_name)
    print('\n[%d/%d] %s' % (weather_index, len(WEATHERS), weather_name))

    owned_actors = []
    paths = create_run_paths(output_root, run_name)
    world.set_weather(weather)
    world.tick()

    try:
        ego = spawn_ego_vehicle(world)
        owned_actors.append(ego)
        ego.set_autopilot(True, traffic_manager.get_port())
        print('    主车：%s' % ego.type_id)

        walkers = spawn_walkers(
            world,
            client,
            args.walkers,
            ego.get_location(),
            args.walker_min_distance,
            args.walker_radius
        )
        owned_actors.extend(walkers)

        cameras, image_queues = attach_cameras(world, ego)
        owned_actors.extend(cameras.values())

        saved = capture_frames(
            world,
            client,
            ego,
            image_queues,
            paths,
            run_name,
            args.frames,
            args,
            owned_actors,
            walkers
        )

        counts = {view: png_count(paths[view]) for view in VIEWS}
        if saved != args.frames or any(
                counts[view] != args.frames for view in VIEWS):
            raise RuntimeError(
                '图片数量校验失败：%s' % ', '.join(
                    '%s=%d' % (view, counts[view]) for view in VIEWS
                )
            )

        print(
            '    本轮完成：STRAIGHT=%d，LEFT=%d，RIGHT=%d' % (
                counts['STRAIGHT'], counts['LEFT'], counts['RIGHT']
            )
        )
        return counts
    finally:
        destroy_actors(client, list(reversed(owned_actors)))


def print_summary(output_root, results, expected_frames):
    print('\n' + '=' * 68)
    print('采集结果')
    all_ok = True
    for weather_name, counts in results:
        line = '  %-16s' % weather_name
        for view in VIEWS:
            count = counts.get(view, 0)
            line += '  %s=%d' % (view, count)
            if count != expected_frames:
                all_ok = False
        print(line)

    expected_total = len(WEATHERS) * len(VIEWS) * expected_frames
    actual_total = sum(sum(counts.values()) for _, counts in results)
    print('  总计：%d/%d 张' % (actual_total, expected_total))
    print('  输出：%s' % output_root)
    print('=' * 68)
    return all_ok and len(results) == len(WEATHERS)


def main():
    args = parse_args()
    random.seed(args.seed)
    output_root = make_output_root(args.output)

    print('=' * 68)
    print('CARLA 多天气采集（v3 稳定增强版）')
    print(
        '6 种天气 x 3 个方向 x 每方向 %d 张 = %d 张' % (
            args.frames, len(WEATHERS) * len(VIEWS) * args.frames
        )
    )
    print(
        '分辨率：%dx%d；拍照间隔：%.1f 秒' % (
            IMG_WIDTH, IMG_HEIGHT, args.capture_interval
        )
    )
    refresh_text = (
        '每 %d 张刷新' % args.walker_refresh
        if args.walker_refresh else '不自动刷新'
    )
    print(
        '随机行人：最多 %d 个，距主车 %g-%g 米，%s' % (
            args.walkers,
            args.walker_min_distance,
            args.walker_radius,
            refresh_text
        )
    )
    print(
        '静止降采样：车速低于 %g km/h 时，每 %d 个时刻保存 1 组' % (
            args.stationary_speed, args.stationary_sample_every
        )
    )
    print('输出目录：%s' % output_root)
    print('=' * 68)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = None
    traffic_manager = None
    original_settings = None
    original_weather = None
    results = []

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        original_settings = world.get_settings()
        original_weather = world.get_weather()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)
            world.set_pedestrians_seed(args.seed)

        print('地图：%s' % world.get_map().name)
        for index, (weather_name, weather) in enumerate(WEATHERS, 1):
            try:
                counts = run_one_weather(
                    world,
                    client,
                    traffic_manager,
                    index,
                    weather_name,
                    weather,
                    output_root,
                    args
                )
            except RuntimeError as error:
                print('    [本轮失败] %s' % error)
                run_name = 'run%02d_%s' % (index, weather_name)
                run_paths = create_run_paths(output_root, run_name)
                counts = {
                    view: png_count(run_paths[view]) for view in VIEWS
                }
            results.append((weather_name, counts))

    except KeyboardInterrupt:
        print('\n用户中止采集。')
    except RuntimeError as error:
        print('\n[错误] %s' % error)
    finally:
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
        if world is not None:
            try:
                if original_weather is not None:
                    world.set_weather(original_weather)
                if original_settings is not None:
                    world.apply_settings(original_settings)
            except RuntimeError:
                pass

    complete = print_summary(output_root, results, args.frames)
    if not complete:
        print('采集未完整完成，请查看上方错误信息和对应天气目录。')
        return 1
    print('全部完成：6 种天气下，每个方向均已保存 %d 张。' % args.frames)
    return 0


if __name__ == '__main__':
    sys.exit(main())
