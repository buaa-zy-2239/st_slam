import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional
import sys
sys.path.insert(0, 'scripts')
from intent_controller import IntentController, ControllerMode


@dataclass
class EpisodeResult:
    episode_id: int
    start_pos: np.ndarray
    goal_pos: np.ndarray
    final_pos: Optional[np.ndarray] = None
    path_length: float = 0.0
    optimal_length: float = 0.0
    success: bool = False
    collision_count: int = 0
    deadlock_count: int = 0
    fallback_count: int = 0
    steps: int = 0
    mode_history: List[str] = field(default_factory=list)


class MockTracker:
    def __init__(self, nodes, edges):
        self.topo_nodes = {i: {"pose": nodes[i]} for i in range(len(nodes))}
        self.edges = edges
        self.current_node = 0
        self.nodes = nodes

    def get_topo_node_pose(self, node_id):
        return self.nodes.get(node_id, [])

    def get_topo_node_neighbors(self, node_id):
        neighbors = []
        for a, b in self.edges:
            if a == node_id:
                neighbors.append(b)
            elif b == node_id:
                neighbors.append(a)
        return neighbors


class SimulatedCostmap:
    def __init__(self, size=80, resolution=0.05):
        self.size = size
        self.resolution = resolution

    def get_costmap(self, robot_pos, obstacles):
        costmap = np.zeros((self.size, self.size), dtype=np.uint8)
        center = self.size // 2

        for obs_pos, radius in obstacles:
            rel_x = (obs_pos[0] - robot_pos[0]) / self.resolution
            rel_z = (obs_pos[2] - robot_pos[2]) / self.resolution

            obs_x = center + int(rel_x)
            obs_z = center + int(rel_z)

            for i in range(max(0, obs_z - radius), min(self.size, obs_z + radius)):
                for j in range(max(0, obs_x - radius), min(self.size, obs_x + radius)):
                    dist = np.sqrt((i - obs_z)**2 + (j - obs_x)**2)
                    if dist < radius:
                        costmap[i, j] = min(255, int(200 * (1 - dist / radius)))

        return costmap


class NavigationSimulator:
    def __init__(self, controller: IntentController):
        self.controller = controller
        self.costmap_gen = SimulatedCostmap()
        self.position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0

    def step(self, cmd):
        angular_steps = max(1, int(abs(cmd.angular_z) / 0.15))
        linear_steps = max(1, int(abs(cmd.linear_x) / 0.05))

        total_rotation = np.sign(cmd.angular_z) * 0.15 * angular_steps
        self.yaw += total_rotation
        self.yaw = self.yaw % (2 * np.pi)

        total_distance = 0.05 * linear_steps * np.sign(cmd.linear_x)
        dx = total_distance * np.sin(self.yaw)
        dz = total_distance * np.cos(self.yaw)
        self.position[0] += dx
        self.position[2] += dz

    def get_costmap(self, obstacles):
        return self.costmap_gen.get_costmap(self.position, obstacles)


def run_episode(
    controller: IntentController,
    tracker: MockTracker,
    sim: NavigationSimulator,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    goal_node_id: int,
    obstacles: list,
    max_steps: int = 300,
    goal_tolerance: float = 0.3,
) -> EpisodeResult:
    sim.position = start_pos.copy()
    sim.yaw = 0.0

    result = EpisodeResult(
            episode_id=0,
            start_pos=start_pos.copy(),
            goal_pos=goal_pos.copy(),
            optimal_length=np.linalg.norm(goal_pos[[0, 2]] - start_pos[[0, 2]]),
        )

    controller.reset()
    controller.set_goal_node(goal_node_id)
    controller.set_current_node(0)

    path_length = 0.0
    last_pos = start_pos.copy()

    for step in range(max_steps):
        controller.set_current_node(tracker.current_node)

        costmap = sim.get_costmap(obstacles)

        cmd, debug = controller.compute_cmd_vel(
            tracker, sim.position, sim.yaw, costmap, return_debug=True
        )

        if debug:
            result.mode_history.append(debug.mode.value)
            if debug.mode == ControllerMode.DEADLOCK_RECOVERY:
                result.deadlock_count += 1
            if debug.mode == ControllerMode.GRAPH_FALLBACK:
                result.fallback_count += 1

        sim.step(cmd)

        step_dist = np.linalg.norm(sim.position - last_pos)
        path_length += step_dist
        last_pos = sim.position.copy()

        collision = np.mean(costmap) > 200
        if collision:
            result.collision_count += 1

        dist_to_goal = np.linalg.norm(sim.position[[0, 2]] - goal_pos[[0, 2]])
        if dist_to_goal < goal_tolerance:
            result.success = True
            result.final_pos = sim.position.copy()
            result.path_length = path_length
            result.steps = step + 1
            return result

        current_pos_2d = sim.position[[0, 2]]

        best_node = tracker.current_node
        best_dist = float('inf')
        for node_id in range(len(tracker.nodes)):
            node_pos = np.array(tracker.nodes.get(node_id, [0, 0, 0, 0, 0, 0, 1])[:3])
            node_pos_2d = node_pos[[0, 2]]
            dist = np.linalg.norm(current_pos_2d - node_pos_2d)

            if node_id != tracker.current_node:
                current_node_pos = np.array(tracker.nodes.get(tracker.current_node, [0, 0, 0, 0, 0, 0, 1])[:3])
                current_node_pos_2d = current_node_pos[[0, 2]]
                mid_point = (current_pos_2d + node_pos_2d) / 2

                if np.linalg.norm(current_pos_2d - mid_point) < np.linalg.norm(current_pos_2d - current_node_pos_2d):
                    if dist < best_dist:
                        best_dist = dist
                        best_node = node_id

        if best_node != tracker.current_node:
            tracker.current_node = best_node
            controller.set_current_node(best_node)

        result.steps = step + 1

    result.final_pos = sim.position.copy()
    result.path_length = path_length
    return result


def create_scenarios():
    nodes = {
        0: [0.0, 0.0, 0.0, 0, 0, 0, 1],
        1: [0.0, 0.0, 1.0, 0, 0, 0, 1],
        2: [0.0, 0.0, 2.0, 0, 0, 0, 1],
        3: [0.0, 0.0, 3.0, 0, 0, 0, 1],
        4: [0.0, 0.0, 4.0, 0, 0, 0, 1],
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

    scenarios = [
        {
            "name": "直线导航 (Straight Line)",
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": np.array([0.0, 0.0, 3.5]),
            "goal_node": 4,
            "obstacles": [],
        },
        {
            "name": "简单绕障 (Simple Obstacle)",
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": np.array([0.0, 0.0, 3.5]),
            "goal_node": 4,
            "obstacles": [
                ([0.0, 0.0, 1.2], 4),
            ],
        },
        {
            "name": "侧向偏移 (Lateral Offset)",
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": np.array([0.0, 0.0, 3.0]),
            "goal_node": 3,
            "obstacles": [
                ([0.0, 0.0, 1.5], 4),
            ],
        },
        {
            "name": "连续节点导航 (Multi-Hop)",
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": np.array([0.0, 0.0, 2.5]),
            "goal_node": 3,
            "obstacles": [],
        },
    ]

    return nodes, edges, scenarios


def evaluate():
    print("=" * 70)
    print("   st_slam 5.0 导航准确率评估 (模拟环境)")
    print("=" * 70)
    print()

    nodes, edges, scenarios = create_scenarios()
    tracker = MockTracker(nodes, edges)

    results = []

    for i, scenario in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] {scenario['name']}")

        controller = IntentController(
            max_linear_speed=0.3,
            max_angular_speed=1.5,
            goal_tolerance=0.3,
            deadlock_frames=5,
            recovery_rotation=0.6,
        )

        sim = NavigationSimulator(controller)

        start_time = time.time()

        result = run_episode(
            controller=controller,
            tracker=tracker,
            sim=sim,
            start_pos=scenario["start"],
            goal_pos=scenario["goal"],
            goal_node_id=scenario["goal_node"],
            obstacles=scenario["obstacles"],
            max_steps=300,
            goal_tolerance=0.3,
        )

        elapsed = time.time() - start_time
        results.append(result)

        status = "SUCCESS ✅" if result.success else "FAILURE ❌"
        if result.path_length > 0.01:
            eff = min(1.0, result.optimal_length / result.path_length)
        else:
            eff = 0

        print(f"    → {status} | Path: {result.path_length:.2f}m | "
              f"Optimal: {result.optimal_length:.2f}m | Eff: {eff*100:.0f}% | "
              f"Deadlocks: {result.deadlock_count} | Time: {elapsed:.2f}s")

    print()
    print("=" * 70)
    print("   聚合准确率指标")
    print("=" * 70)
    print()

    num_episodes = len(results)
    success_count = sum(1 for r in results if r.success)
    success_rate = success_count / num_episodes

    total_path = sum(r.path_length for r in results)
    total_optimal = sum(r.optimal_length for r in results)

    spl_sum = sum(
        min(1.0, r.optimal_length / r.path_length) if r.path_length > 0.01 else 0
        for r in results
    )
    spl = spl_sum / num_episodes

    avg_dist_to_goal = sum(
        np.linalg.norm(r.final_pos[:2] - r.goal_pos[:2]) if r.final_pos is not None else 10
        for r in results
    ) / num_episodes

    total_collisions = sum(r.collision_count for r in results)
    total_steps = sum(r.steps for r in results)
    collision_rate = total_collisions / total_steps if total_steps > 0 else 0

    deadlock_eps = sum(1 for r in results if r.deadlock_count > 0)
    deadlock_rate = deadlock_eps / num_episodes

    fallback_eps = sum(1 for r in results if r.fallback_count > 0)
    fallback_rate = fallback_eps / num_episodes

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    核心准确率指标                           │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Success Rate (SR):     {success_rate*100:6.1f} %                          │")
    print(f"  │  SPL (Path Efficiency):  {spl*100:6.1f} %                          │")
    print(f"  │  Avg Path Length:       {total_path/num_episodes:6.2f} m                       │")
    print(f"  │  Avg Optimal Length:    {total_optimal/num_episodes:6.2f} m                       │")
    print(f"  │  Avg Distance to Goal:  {avg_dist_to_goal:6.2f} m                       │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    安全性指标                               │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Collision Rate:       {collision_rate*100:6.1f} %                          │")
    print(f"  │  Deadlock Rate:        {deadlock_rate*100:6.1f} %                          │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    降级机制统计                               │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Fallback Rate:        {fallback_rate*100:6.1f} %                          │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    print("  " + "=" * 68)
    print("  │                    性能参考指标                             │")
    print("  " + "=" * 68)
    print()
    print("  参考指标 (来自 FPS Benchmark):")
    print("    • Full Loop FPS:    ~350 FPS  (目标: 90+)")
    print("    • IntentController: ~0.4 ms   (目标: <2ms)")
    print()

    sr_grade = "A" if success_rate >= 0.75 else "B" if success_rate >= 0.5 else "C" if success_rate >= 0.25 else "F"
    spl_grade = "A" if spl >= 0.5 else "B" if spl >= 0.3 else "C" if spl >= 0.1 else "F"
    coll_grade = "A" if collision_rate < 0.05 else "B" if collision_rate < 0.15 else "C" if collision_rate < 0.3 else "F"
    dlock_grade = "A" if deadlock_rate < 0.1 else "B" if deadlock_rate < 0.3 else "C" if deadlock_rate < 0.5 else "F"

    print("  [评分标准]")
    print("    SR (Success Rate):      75%+ 优秀 | 50%+ 良好 | 25%+ 合格 | <25% 需改进")
    print("    SPL (Path Efficiency): 50%+ 优秀 | 30%+ 良好 | 10%+ 合格 | <10% 需改进")
    print()
    print("  [分项评分]")
    print(f"    Success Rate:      {sr_grade} ({success_rate*100:.0f}%)")
    print(f"    Path Efficiency:   {spl_grade} ({spl*100:.0f}%)")
    print(f"    Collision Safety:  {coll_grade} ({collision_rate*100:.0f}%)")
    print(f"    Deadlock Defense:  {dlock_grade} ({deadlock_rate*100:.0f}%)")

    grades = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    overall = (grades[sr_grade] + grades[spl_grade] + grades[coll_grade] + grades[dlock_grade]) / 4
    overall_grade = "A" if overall >= 4.5 else "B" if overall >= 3.5 else "C" if overall >= 2.5 else "D" if overall >= 1.5 else "F"

    print("    " + "-" * 40)
    print(f"    Overall Grade:     {overall_grade} ({overall:.1f}/5.0)")
    print()
    print("=" * 70)

    return {
        "success_rate": success_rate,
        "spl": spl,
        "collision_rate": collision_rate,
        "deadlock_rate": deadlock_rate,
        "overall": overall,
    }


if __name__ == "__main__":
    evaluate()
