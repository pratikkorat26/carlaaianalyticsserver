import carla
import os
import time
import math
from datetime import datetime
from random import shuffle

def save_camera_feed_for_duration(camera, output_dir="carla_images", duration_sec=120):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    def process_image(image):
        if time.time() - start_time > duration_sec:
            camera.stop()
            print("✅ Image capture complete.")
            return
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        image_path = os.path.join(output_dir, f"{timestamp}.jpg")
        image.save_to_disk(image_path)

    camera.listen(lambda image: process_image(image))

def get_angle_between_vectors(v1, v2):
    dot = v1.x * v2.x + v1.y * v2.y
    det = v1.x * v2.y - v1.y * v2.x
    return math.atan2(det, dot)

def drive_through_waypoints(vehicle, waypoint_list, duration=300):
    start_time = time.time()
    idx = 0

    while time.time() - start_time < duration:
        if idx >= len(waypoint_list):
            print("🚗 No more waypoints to follow.")
            break

        target_wp = waypoint_list[idx]
        vehicle_loc = vehicle.get_location()
        vehicle_rot = vehicle.get_transform().rotation
        vehicle_yaw = math.radians(vehicle_rot.yaw)

        direction = carla.Vector3D(
            x=target_wp.transform.location.x - vehicle_loc.x,
            y=target_wp.transform.location.y - vehicle_loc.y
        )
        distance = math.sqrt(direction.x**2 + direction.y**2)

        if distance < 2.5:
            idx += 1
            continue

        forward = carla.Vector3D(math.cos(vehicle_yaw), math.sin(vehicle_yaw))
        angle = get_angle_between_vectors(forward, direction)

        control = carla.VehicleControl()
        control.throttle = 0.4
        control.steer = max(min(angle, 1.0), -1.0)
        vehicle.apply_control(control)
        time.sleep(0.05)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Disable rendering
    settings = world.get_settings()
    settings.no_rendering_mode = True
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]

    # Try multiple spawn points
    spawn_points = world.get_map().get_spawn_points()
    shuffle(spawn_points)  # for variety
    vehicle = None
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            break

    if not vehicle:
        print("❌ No free spawn points for vehicle.")
        return

    # Disable autopilot
    vehicle.set_autopilot(False)

    # Attach RGB camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', '0.1')  # Controls frame rate (10 FPS)
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Save camera feed
    save_camera_feed_for_duration(camera, output_dir="carla_images", duration_sec=120)

    # Generate diverse path
    current_wp = world.get_map().get_waypoint(vehicle.get_location())
    waypoints = [current_wp]
    for _ in range(50):  # generate 50 ahead
        next_wp = waypoints[-1].next(15.0)
        if next_wp:
            waypoints.append(next_wp[0])
        else:
            break

    # Start custom drive
    drive_through_waypoints(vehicle, waypoints, duration=120)

    # Cleanup
    camera.destroy()
    vehicle.destroy()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
