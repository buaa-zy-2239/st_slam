import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class ControllerMode(Enum):
    NORMAL = "normal"
    DEADLOCK_RECOVERY = "deadlock_recovery"
    GRAPH_FALLBACK = "graph_fallback"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"


@dataclass
class CmdVel:
    linear_x: float
    angular_z: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.linear_x, self.angular_z)


@dataclass
class DebugInfo:
    mode: ControllerMode
    intent_vector: np.ndarray
    repulsion_vector: np.ndarray
    deadlock_counter: int
    fallback_active: bool


class IntentController:
    def __init__(
        self,
        max_linear_speed: float = 0.3,
        max_angular_speed: float = 1.5,
        costmap_resolution: float = 0.05,
        costmap_size: int = 80,
        goal_tolerance: float = 0.3,
        obstacle_threshold: float = 200.0,
        repulsion_gain: float = 2.0,
        intent_gain: float = 1.0,
        deadlock_threshold: float = 0.02,
        deadlock_frames: int = 5,
        recovery_rotation: float = 0.6,
        min_repulsion_range: float = 0.3,
        max_repulsion_range: float = 1.5,
        speed_adaptive_gain: bool = True,
    ):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.costmap_resolution = costmap_resolution
        self.costmap_size = costmap_size
        self.goal_tolerance = goal_tolerance
        self.base_obstacle_threshold = obstacle_threshold
        self.obstacle_threshold = obstacle_threshold
        self.base_repulsion_gain = repulsion_gain
        self.repulsion_gain = repulsion_gain
        self.intent_gain = intent_gain

        self.deadlock_threshold = deadlock_threshold
        self.deadlock_frames = deadlock_frames
        self.deadlock_counter = 0
        self.recovery_rotation = recovery_rotation
        self.recovery_direction = 1.0

        self.min_repulsion_range = min_repulsion_range
        self.max_repulsion_range = max_repulsion_range
        self.speed_adaptive_gain = speed_adaptive_gain

        self.goal_node_id: Optional[int] = None
        self.current_node_id: int = 0
        self.last_linear_x: float = 0.0
        self.mode = ControllerMode.NORMAL

        self.tangent_force_ratio: float = 1.2
        self.front_blocked_threshold: float = 30.0
        self.avoidance_direction: int = 1
        self.front_cost_history: List[float] = []

    def compute_2hop_intent_vector(
        self,
        tracker,
        current_pos: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        if self.goal_node_id is None:
            return np.zeros(2), False

        goal_pos = self._get_node_position(tracker, self.goal_node_id)
        if goal_pos is None:
            return np.zeros(2), True

        neighbors = tracker.get_topo_node_neighbors(self.current_node_id)

        if not neighbors:
            intent = goal_pos[[0, 2]] - current_pos[[0, 2]]
            dist = np.linalg.norm(intent)
            if dist > 0.01:
                intent /= dist
            return intent, False

        for hop1_id in neighbors:
            hop1_pos = self._get_node_position(tracker, hop1_id)
            if hop1_pos is None:
                continue

            hop1_vec = hop1_pos[[0, 2]] - current_pos[[0, 2]]
            hop1_dist = np.linalg.norm(hop1_vec)

            if hop1_id == self.goal_node_id:
                return hop1_vec / (hop1_dist + 1e-6) if hop1_dist > 0.01 else np.zeros(2), False

        best_2hop = None
        best_score = -float("inf")

        for hop1_id in neighbors:
            hop1_pos = self._get_node_position(tracker, hop1_id)
            if hop1_pos is None:
                continue

            hop1_vec = hop1_pos[[0, 2]] - current_pos[[0, 2]]
            hop1_dist = np.linalg.norm(hop1_vec)

            hop2_neighbors = tracker.get_topo_node_neighbors(hop1_id)

            for hop2_id in hop2_neighbors:
                if hop2_id == self.current_node_id:
                    continue

                hop2_pos = self._get_node_position(tracker, hop2_id)
                if hop2_pos is None:
                    continue

                hop2_vec = hop2_pos[[0, 2]] - hop1_pos[[0, 2]]
                hop2_dist = np.linalg.norm(hop2_vec)

                goal_vec = goal_pos[[0, 2]] - hop1_pos[[0, 2]]
                goal_dist = np.linalg.norm(goal_vec)

                alignment = np.dot(hop2_vec, goal_vec) / (hop2_dist * goal_dist + 1e-6)
                score = alignment / (hop1_dist + 1.0)

                if score > best_score:
                    best_score = score
                    best_2hop = hop1_vec / (hop1_dist + 1e-6) if hop1_dist > 0.01 else np.zeros(2)

        if best_2hop is not None:
            return best_2hop, False

        return np.zeros(2), True

    def _compute_fallback_intent(
        self,
        tracker,
        current_pos: np.ndarray,
    ) -> np.ndarray:
        if self.goal_node_id is None:
            return np.zeros(2)

        goal_pos = self._get_node_position(tracker, self.goal_node_id)
        if goal_pos is None:
            return np.zeros(2)

        intent = goal_pos[[0, 2]] - current_pos[[0, 2]]
        dist = np.linalg.norm(intent)
        if dist > 0.01:
            intent /= dist
        return intent

    def _get_node_position(self, tracker, node_id: int) -> Optional[np.ndarray]:
        pose = tracker.get_topo_node_pose(node_id)
        if not pose:
            return None
        return np.array(pose[:3])

    def _update_speed_adaptive_parameters(self, current_linear_speed: float):
        if not self.speed_adaptive_gain:
            return

        speed_ratio = abs(current_linear_speed) / (self.max_linear_speed + 1e-6)

        self.repulsion_gain = self.base_repulsion_gain * (0.5 + 0.5 * (1.0 - speed_ratio))

        if speed_ratio > 0.7:
            self.obstacle_threshold = self.base_obstacle_threshold * 0.7
        elif speed_ratio < 0.2:
            self.obstacle_threshold = self.base_obstacle_threshold * 1.2
        else:
            self.obstacle_threshold = self.base_obstacle_threshold

    def compute_apf_repulsion(
        self,
        costmap: np.ndarray,
        current_yaw: float,
        repulsion_range: Optional[float] = None,
    ) -> np.ndarray:
        if repulsion_range is None:
            repulsion_range = self.max_repulsion_range

        effective_threshold = self.obstacle_threshold * 0.8

        obstacle_mask = costmap > effective_threshold
        if not np.any(obstacle_mask):
            return np.zeros(2)

        obstacle_coords = np.argwhere(obstacle_mask)
        if len(obstacle_coords) == 0:
            return np.zeros(2)

        center = self.costmap_size // 2

        obstacle_x = (obstacle_coords[:, 1] - center) * self.costmap_resolution
        obstacle_y = (center - obstacle_coords[:, 0]) * self.costmap_resolution

        dist = np.sqrt(obstacle_x**2 + obstacle_y**2)
        dist = np.maximum(dist, 0.05)

        range_mask = dist <= repulsion_range
        if not np.any(range_mask):
            return np.zeros(2)

        obstacle_x = obstacle_x[range_mask]
        obstacle_y = obstacle_y[range_mask]
        dist = dist[range_mask]

        if len(dist) == 0:
            return np.zeros(2)

        costs = costmap[obstacle_mask][range_mask].astype(np.float32) / 255.0
        strengths = costs * self.repulsion_gain / (dist**2)

        rep_x = -obstacle_x / dist * strengths
        rep_y = -obstacle_y / dist * strengths

        repulsion = np.array([rep_x.sum(), rep_y.sum()])

        tangent_force = self._compute_tangent_force(obstacle_x, obstacle_y, dist, strengths)

        max_repulsion = 2.5
        repulsion_norm = np.linalg.norm(repulsion)
        if repulsion_norm > max_repulsion:
            repulsion = repulsion / repulsion_norm * max_repulsion

        return repulsion + tangent_force

    def _compute_tangent_force(
        self,
        obstacle_x: np.ndarray,
        obstacle_y: np.ndarray,
        dist: np.ndarray,
        strengths: np.ndarray,
    ) -> np.ndarray:
        if len(obstacle_x) == 0:
            return np.zeros(2)

        tangent_x = -obstacle_y / (dist + 0.05)
        tangent_y = obstacle_x / (dist + 0.05)

        tangent_strengths = strengths * self.tangent_force_ratio

        tangent_force = np.array([
            (tangent_x * tangent_strengths).sum(),
            (tangent_y * tangent_strengths).sum(),
        ])

        max_tangent = 1.5
        tangent_norm = np.linalg.norm(tangent_force)
        if tangent_norm > max_tangent:
            tangent_force = tangent_force / tangent_norm * max_tangent

        return tangent_force

    def _detect_deadlock(
        self,
        forward_component: float,
        repulsion_norm: float,
    ) -> bool:
        in_deadlock_zone = forward_component < 0.05
        in_high_repulsion = repulsion_norm > 1.0
        has_intent = self.goal_node_id is not None

        if in_deadlock_zone and in_high_repulsion and has_intent:
            self.deadlock_counter += 1
        else:
            self.deadlock_counter = 0

        return self.deadlock_counter >= self.deadlock_frames

    def _apply_deadlock_recovery(self, angular_z: float) -> float:
        self.mode = ControllerMode.DEADLOCK_RECOVERY

        self.deadlock_counter = 0

        perturbation = self.recovery_rotation * self.recovery_direction

        self.recovery_direction *= -1.0

        return angular_z + perturbation

    def compute_cmd_vel(
        self,
        tracker,
        current_pos: np.ndarray,
        current_yaw: float,
        costmap: np.ndarray,
        return_debug: bool = False,
    ) -> Tuple[CmdVel, Optional[DebugInfo]]:
        self._update_speed_adaptive_parameters(self.last_linear_x)

        intent, graph_broken = self.compute_2hop_intent_vector(tracker, current_pos)

        if graph_broken and self.mode != ControllerMode.GRAPH_FALLBACK:
            self.mode = ControllerMode.GRAPH_FALLBACK
            intent = self._compute_fallback_intent(tracker, current_pos)

        if self.mode == ControllerMode.GRAPH_FALLBACK and not graph_broken:
            self.mode = ControllerMode.NORMAL

        repulsion = self.compute_apf_repulsion(costmap, current_yaw)

        intent_weight = self.intent_gain
        if self.mode == ControllerMode.OBSTACLE_AVOIDANCE:
            intent_weight = self.intent_gain * 0.15

        combined = intent_weight * intent + repulsion

        if self.goal_node_id is not None:
            goal_pos = self._get_node_position(tracker, self.goal_node_id)
            if goal_pos is not None:
                dist_to_goal = np.linalg.norm(goal_pos[[0, 2]] - current_pos[[0, 2]])
                if dist_to_goal < self.goal_tolerance:
                    self.goal_node_id = None
                    self.mode = ControllerMode.NORMAL
                    if return_debug:
                        return CmdVel(linear_x=0.0, angular_z=0.0), DebugInfo(
                            mode=self.mode,
                            intent_vector=intent,
                            repulsion_vector=repulsion,
                            deadlock_counter=self.deadlock_counter,
                            fallback_active=self.mode == ControllerMode.GRAPH_FALLBACK,
                        )
                    return CmdVel(linear_x=0.0, angular_z=0.0), None

        cmd = self._compute_base_cmd(intent, repulsion, current_yaw)

        front_blocked = self._check_front_blocked(costmap, current_yaw)
        if front_blocked:
            cmd = self._apply_avoidance_cmd(cmd, repulsion)

        if self._detect_deadlock(cmd.linear_x, np.linalg.norm(repulsion)):
            cmd.angular_z = self._apply_deadlock_recovery(cmd.angular_z)
        else:
            if self.mode == ControllerMode.OBSTACLE_AVOIDANCE and not front_blocked:
                self.mode = ControllerMode.NORMAL

        self.last_linear_x = cmd.linear_x

        if return_debug:
            debug_info = DebugInfo(
                mode=self.mode,
                intent_vector=intent.copy(),
                repulsion_vector=repulsion.copy(),
                deadlock_counter=self.deadlock_counter,
                fallback_active=self.mode == ControllerMode.GRAPH_FALLBACK,
            )
            return cmd, debug_info

        return cmd, None

    def _compute_base_cmd(
        self,
        intent: np.ndarray,
        repulsion: np.ndarray,
        current_yaw: float,
    ) -> CmdVel:
        combined = self.intent_gain * intent + repulsion

        robot_forward = np.array([np.sin(current_yaw), np.cos(current_yaw)])
        robot_left = np.array([np.cos(current_yaw), -np.sin(current_yaw)])

        forward_component = np.dot(combined, robot_forward)
        lateral_component = np.dot(combined, robot_left)

        linear_x = np.clip(forward_component, -self.max_linear_speed, self.max_linear_speed)

        angular_z = np.clip(lateral_component * 3.0, -self.max_angular_speed, self.max_angular_speed)

        if forward_component < 0:
            angular_z = np.clip(angular_z * 2.0, -self.max_angular_speed, self.max_angular_speed)

        return CmdVel(linear_x=linear_x, angular_z=angular_z)

    def _check_front_blocked(self, costmap: np.ndarray, current_yaw: float) -> bool:
        center = self.costmap_size // 2
        front_range = int(0.5 / self.costmap_resolution)

        front_min_row = center - front_range
        front_max_row = center + front_range
        front_cols = range(max(0, center - 15), min(self.costmap_size, center + 15))

        if front_min_row < 0 or front_max_row >= self.costmap_size:
            return False

        front_cost = costmap[front_min_row:front_max_row, front_cols].mean()

        self.front_cost_history.append(front_cost)
        if len(self.front_cost_history) > 3:
            self.front_cost_history.pop(0)

        avg_front_cost = np.mean(self.front_cost_history) if self.front_cost_history else 0

        return avg_front_cost > self.front_blocked_threshold

    def _apply_avoidance_cmd(self, cmd: CmdVel, repulsion: np.ndarray) -> CmdVel:
        self.mode = ControllerMode.OBSTACLE_AVOIDANCE

        if repulsion[1] < -0.5:
            angular_bias = self.max_angular_speed * 0.95
        elif repulsion[1] > 0.5:
            angular_bias = -self.max_angular_speed * 0.95
        else:
            angular_bias = self.avoidance_direction * self.max_angular_speed * 0.95

        cmd.angular_z = np.clip(angular_bias, -self.max_angular_speed, self.max_angular_speed)
        cmd.linear_x = np.clip(self.max_linear_speed * 0.5, 0.1, self.max_linear_speed * 0.6)

        return cmd

    def reset(self):
        self.goal_node_id = None
        self.current_node_id = 0
        self.deadlock_counter = 0
        self.mode = ControllerMode.NORMAL
        self.last_linear_x = 0.0

    def set_goal_by_label(self, tracker, label: str, min_confidence: float = 0.5) -> bool:
        candidate_nodes = tracker.find_nodes_by_label(label, min_confidence)
        if not candidate_nodes:
            return False
        self.goal_node_id = candidate_nodes[0]
        self.mode = ControllerMode.NORMAL
        return True

    def set_goal_node(self, node_id: int) -> None:
        self.goal_node_id = node_id
        self.mode = ControllerMode.NORMAL

    def set_current_node(self, node_id: int) -> None:
        self.current_node_id = node_id

    def has_active_goal(self) -> bool:
        return self.goal_node_id is not None

    def get_mode(self) -> ControllerMode:
        return self.mode


def euler_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def get_current_pose(tracker) -> Tuple[np.ndarray, float]:
    pose = tracker.get_smooth_pose()
    pos = np.array(pose[:3])
    yaw = euler_from_quaternion(pose[3], pose[4], pose[5], pose[6])
    return pos, yaw
