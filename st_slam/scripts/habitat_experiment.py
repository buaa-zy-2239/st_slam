import numpy as np
import st_slam_tracker
import habitat_sim
import time
from intent_controller import IntentController, get_current_pose

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "/tmp/habitat_data/scene_datasets/habitat-test-scenes/apartment_1.glb"
sim_cfg.gpu_device_id = -1
sim_cfg.enable_physics = False
sim_cfg.override_scene_light_defaults = True
sim_cfg.scene_light_setup = "flat_lighting"

rgb_spec = habitat_sim.CameraSensorSpec()
rgb_spec.uuid = "rgb"
rgb_spec.resolution = [480, 640]
rgb_spec.hfov = 90
rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_spec.position = [0.0, 0.0, 0.0]

depth_spec = habitat_sim.CameraSensorSpec()
depth_spec.uuid = "depth"
depth_spec.resolution = [480, 640]
depth_spec.hfov = 90
depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
depth_spec.position = [0.0, 0.0, 0.0]

agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

tracker = st_slam_tracker.Tracker()
controller = IntentController(
    max_linear_speed=0.25,
    max_angular_speed=1.2,
    goal_tolerance=0.4,
)

print("=" * 60)
print("   st_slam 5.0 Intent Controller FPS Benchmark")
print("=" * 60)
print()

obs = sim.get_sensor_observations()
rgb = obs["rgb"][:, :, :3].astype(np.uint8)
depth = obs["depth"].squeeze().astype(np.float32)

print("Initialization frame...")
tracker.step(rgb, depth)

print("Running 500 frames for FPS benchmark...")
print()

NUM_FRAMES = 500
timestep_times = []
controller_times = []
total_start = time.perf_counter()

for step in range(NUM_FRAMES):
    obs = sim.get_sensor_observations()
    rgb = obs["rgb"][:, :, :3].astype(np.uint8)
    depth = obs["depth"].squeeze().astype(np.float32)

    t0 = time.perf_counter()
    tracker.step(rgb, depth)
    t1 = time.perf_counter()

    costmap = tracker.get_local_costmap()
    pos, yaw = get_current_pose(tracker)
    controller.compute_cmd_vel(tracker, pos, yaw, costmap)
    t2 = time.perf_counter()

    timestep_times.append(t1 - t0)
    controller_times.append(t2 - t1)

    agent_state = sim.get_agent(0).get_state()
    agent_state.position += np.array([0.0, 0.0, -0.05])
    sim.get_agent(0).set_state(agent_state)

    if (step + 1) % 100 == 0:
        print(f"  Progress: {step + 1}/{NUM_FRAMES} frames...")

total_elapsed = time.perf_counter() - total_start

timestep_times = np.array(timestep_times)
controller_times = np.array(controller_times)
total_per_frame = timestep_times + controller_times

print()
print("=" * 60)
print("   FPS Performance Report")
print("=" * 60)
print()
print(f"Total Frames:        {NUM_FRAMES}")
print(f"Total Time:          {total_elapsed:.3f} s")
print(f"")
print(f"  [Tracker.step()]")
print(f"    Mean:            {timestep_times.mean()*1000:.2f} ms/frame")
print(f"    Std:             {timestep_times.std()*1000:.2f} ms")
print(f"    Min:             {timestep_times.min()*1000:.2f} ms")
print(f"    Max:             {timestep_times.max()*1000:.2f} ms")
print(f"    FPS:             {1.0/timestep_times.mean():.1f}")
print(f"")
print(f"  [IntentController]")
print(f"    Mean:            {controller_times.mean()*1000:.2f} ms/frame")
print(f"    Std:             {controller_times.std()*1000:.2f} ms")
print(f"    Min:             {controller_times.min()*1000:.2f} ms")
print(f"    Max:             {controller_times.max()*1000:.2f} ms")
print(f"    FPS:             {1.0/controller_times.mean():.1f}")
print(f"")
print(f"  [Combined (Full Loop)]")
print(f"    Mean:            {total_per_frame.mean()*1000:.2f} ms/frame")
print(f"    Std:             {total_per_frame.std()*1000:.2f} ms")
print(f"    FPS:             {1.0/total_per_frame.mean():.1f}")
print(f"    FPS (90+ target): {'PASS' if 1.0/total_per_frame.mean() >= 90 else 'FAIL'}")
print()

print("=" * 60)
print("   SLAM Status")
print("=" * 60)
print(f"KFs:                {tracker.get_num_kfs()}")
print(f"MPs:                {tracker.get_num_mps()}")
print(f"Loops:              {tracker.get_num_loops()}")
print(f"State:              {tracker.get_state()}")
print(f"Topo nodes:         {tracker.get_num_topo_nodes()}")
print(f"Topo edges:         {tracker.get_num_topo_edges()}")
print()

print("=" * 60)
print("   Intent Controller Verification")
print("=" * 60)

for nid in range(min(5, tracker.get_num_topo_nodes())):
    tracker.attach_label(nid, "chair", 0.92, 56)
    if nid % 2 == 0:
        tracker.attach_label(nid, "table", 0.85, 60)

controller.set_goal_by_label(tracker, "table", 0.5)
print(f"Goal node:          {controller.goal_node_id}")
print(f"Goal active:        {controller.has_active_goal()}")
print(f"Costmap:            {tracker.get_local_costmap().shape}")

pos, yaw = get_current_pose(tracker)
cmd, _ = controller.compute_cmd_vel(tracker, pos, yaw, tracker.get_local_costmap())
print(f"cmd_vel:            linear={cmd.linear_x:.3f}, angular={cmd.angular_z:.3f}")
print(f"Controller mode:    {controller.get_mode().value}")
print()

print("=" * 60)
print("   BENCHMARK COMPLETE")
print("=" * 60)

sim.close()
