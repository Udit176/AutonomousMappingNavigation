#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import heapq
import numpy as np
from math import sqrt, atan2, ceil
import time

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class MapProcessor:
    def __init__(self):
        self.map_array = None
        self.inf_map_img_array = None
        self.map_graph = {}
        self.map_resolution = 0.05
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

    def load_map_from_occupancy_grid(self, occupancy_grid):
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height
        self.map_resolution = occupancy_grid.info.resolution
        self.map_origin_x = occupancy_grid.info.origin.position.x
        self.map_origin_y = occupancy_grid.info.origin.position.y

        self.map_array = np.array(occupancy_grid.data).reshape((height, width))
        self.map_array = np.where(self.map_array == -1, 100, self.map_array)
        self.inf_map_img_array = np.zeros_like(self.map_array)

    def __inflate_obstacle(self, kernel, map_array, i, j):
        dx = kernel.shape[0] // 2
        dy = kernel.shape[1] // 2
        for k in range(i - dx, i + dx + 1):
            for l in range(j - dy, j + dy + 1):
                if 0 <= k < map_array.shape[0] and 0 <= l < map_array.shape[1]:
                    map_array[k, l] += kernel[k - i + dx, l - j + dy]

    def inflate_map(self, kernel):
        for i in range(self.map_array.shape[0]):
            for j in range(self.map_array.shape[1]):
                if self.map_array[i, j] > 50:
                    self.__inflate_obstacle(kernel, self.inf_map_img_array, i, j)

        range_value = self.inf_map_img_array.max() - self.inf_map_img_array.min()
        if range_value != 0:
            self.inf_map_img_array = (self.inf_map_img_array - self.inf_map_img_array.min()) / range_value
        else:
            self.inf_map_img_array.fill(0)

    def get_graph_from_map(self, logger=None):
        self.map_graph = {}  # Reset the graph
        free_cells = 0
        for i in range(self.inf_map_img_array.shape[0]):
            for j in range(self.inf_map_img_array.shape[1]):
                if self.inf_map_img_array[i, j] == 0:
                    self.map_graph[f"{i},{j}"] = {"children": {}, "coords": (i, j)}
                    free_cells +=1
        if logger:
            logger.info(f"Number of free cells added to graph: {free_cells}")

        connections = 0
        for node_name, node_data in self.map_graph.items():
            i, j = node_data["coords"]
            # 8-connected grid
            for di, dj, weight in [
                (-1,0,1), (1,0,1), (0,-1,1), (0,1,1),
                (-1,-1,math.sqrt(2)), (-1,1,math.sqrt(2)),
                (1,-1,math.sqrt(2)), (1,1,math.sqrt(2))
            ]:
                ni, nj = i+di, j+dj
                neighbor_name = f"{ni},{nj}"
                if neighbor_name in self.map_graph:
                    self.map_graph[node_name]["children"][neighbor_name] = weight
                    connections +=1
        if logger:
            logger.info(f"Number of connections in graph: {connections}")

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1/(2.0*math.pi*sigma**2)
        return np.exp(-((x**2+y**2)/(2.0*sigma**2)))*normal

    def world_to_index(self, wx, wy):
        mx = int((wx - self.map_origin_x)/self.map_resolution)
        my = int((wy - self.map_origin_y)/self.map_resolution)
        if mx < 0 or my < 0 or mx >= self.map_array.shape[1] or my >= self.map_array.shape[0]:
            return None
        return (my, mx)

    def index_to_world(self, i, j):
        wx = j*self.map_resolution + self.map_origin_x
        wy = i*self.map_resolution + self.map_origin_y
        return (wx, wy)

    def is_free_cell(self, i, j):
        if i<0 or j<0 or i>=self.inf_map_img_array.shape[0] or j>=self.inf_map_img_array.shape[1]:
            return False
        return self.inf_map_img_array[i, j] == 0


class AStar:
    def __init__(self, graph, logger=None):
        self.graph = graph
        self.logger = logger

    def solve(self, start, end):
        if start not in self.graph or end not in self.graph:
            if self.logger:
                self.logger.warn("Start or end node not in graph.")
            return [], float('inf')

        open_set = []
        heapq.heappush(open_set, (0, start))
        g_scores = {node: float("inf") for node in self.graph}
        g_scores[start] = 0
        came_from = {}
        closed_set = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == end:
                if self.logger:
                    self.logger.info(f"A* found path with cost {g_scores[end]}")
                return self.reconstruct_path(came_from, current), g_scores[end]

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor, weight in self.graph[current]["children"].items():
                if neighbor in closed_set:
                    continue
                tentative_g = g_scores[current] + weight
                if tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
                    if self.logger:
                        self.logger.debug(f"Updating node {neighbor} with g_score={tentative_g:.2f} and f_score={f_score:.2f}")

        if self.logger:
            self.logger.warn("A* failed to find a path.")
        return [], float('inf')

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        if self.logger:
            self.logger.debug(f"Reconstructed path: {path}")
        return path[::-1]

    def heuristic(self, node, goal):
        node_i, node_j = map(int, node.split(","))
        goal_i, goal_j = map(int, goal.split(","))
        dx = abs(goal_i - node_i)
        dy = abs(goal_j - node_j)
        F = math.sqrt(2) - 1
        if dx < dy:
            return F * dx + dy
        else:
            return F * dy + dx


class Navigation(Node):
    def __init__(self, node_name="Navigation"):
        super().__init__(node_name)
        self.map_processor = MapProcessor()
        self.a_star_solver = None
        self.linear_speed = 1.0
        self.angular_speed = 0.3
        self.goal_tolerance = 0.1

        self.path_pub = self.create_publisher(Path, "global_plan", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # Publishers for debugging
        self.original_map_pub = self.create_publisher(OccupancyGrid, "original_map", 10)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, "inflated_map", 10)
        self.marker_pub = self.create_publisher(Marker, "path_markers", 10)

        self.goal_pose = None
        self.ttbot_pose = None
        self.map_received = False
        self.path = []
        self.current_waypoint_idx = 0

        # Flag to indicate obstacle detection
        self.obstacle_detected = False

        # Parameters for obstacle detection
        self.obstacle_distance_threshold = 0.5  # meters
        self.obstacle_angle_range = 30  # degrees

        #DO NOT REMOVE
        qos_profile_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_pose = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_goal = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_profile_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriptions
        self.create_subscription(OccupancyGrid, "/map", self.map_cbk, qos_profile_map)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.__ttbot_pose_cbk, qos_profile_pose)
        self.create_subscription(PoseStamped, "/move_base_simple/goal", self.__goal_pose_cbk, qos_profile_goal)
        self.create_subscription(LaserScan, "/scan", self.__laser_scan_cbk, qos_profile_scan)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Navigation node initialized with A* and obstacle detection.")

    def map_cbk(self, data):
        self.map_processor.load_map_from_occupancy_grid(data)
        kernel = self.map_processor.gaussian_kernel(7, sigma=1.5)
        self.map_processor.inflate_map(kernel)
        self.map_processor.get_graph_from_map(self.get_logger())
        self.a_star_solver = AStar(self.map_processor.map_graph, self.get_logger())
        self.map_received = True
        self.get_logger().info("Map received and processed.")

        # Publish original map
        self.original_map_pub.publish(data)

        # Publish inflated map
        inflated_map = OccupancyGrid()
        inflated_map.header = data.header
        inflated_map.info = data.info
        # Convert inf_map_img_array to list and scale appropriately
        inflated_map.data = (self.map_processor.inf_map_img_array * 100).astype(int).flatten().tolist()
        self.inflated_map_pub.publish(inflated_map)

    def __ttbot_pose_cbk(self, data):
        self.ttbot_pose = data.pose.pose
        # Calculate yaw from quaternion
        q = self.ttbot_pose.orientation
        siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.get_logger().debug(f"Robot pose: x={self.ttbot_pose.position.x:.2f}, y={self.ttbot_pose.position.y:.2f}, yaw={math.degrees(yaw):.2f}")

    def __goal_pose_cbk(self, data):
        self.goal_pose = data
        self.get_logger().info(f"Goal pose received: x={data.pose.position.x:.2f}, y={data.pose.position.y:.2f}")
        if self.map_received and self.ttbot_pose is not None:
            self.plan_path()
        else:
            self.get_logger().warn("Cannot plan yet: Map or pose not ready.")

    def plan_path(self):
        start_x = self.ttbot_pose.position.x
        start_y = self.ttbot_pose.position.y
        end_x = self.goal_pose.pose.position.x
        end_y = self.goal_pose.pose.position.y

        start_idx = self.map_processor.world_to_index(start_x, start_y)
        end_idx = self.map_processor.world_to_index(end_x, end_y)

        self.get_logger().info(f"Start Position: x={start_x}, y={start_y} => Index={start_idx}")
        self.get_logger().info(f"Goal Position: x={end_x}, y={end_y} => Index={end_idx}")

        if start_idx is None or end_idx is None:
            self.get_logger().error("Start or goal out of map bounds. No path can be found.")
            self.path = []
            return

        start_node = f"{start_idx[0]},{start_idx[1]}"
        end_node = f"{end_idx[0]},{end_idx[1]}"

        if end_node not in self.map_processor.map_graph:
            self.get_logger().error("Goal cell is not free or not in graph. Can't plan.")
            self.path = []
            return

        self.get_logger().info(f"Planning path from {start_node} to {end_node} using A*.")
        path, cost = self.a_star_solver.solve(start_node, end_node)
        if not path:
            self.get_logger().warn("No path found.")
            self.path = []
            return

        world_path = []
        for node in path:
            i, j = map(int, node.split(","))
            wx, wy = self.map_processor.index_to_world(i, j)
            world_path.append((wx, wy))

        self.path = world_path
        self.current_waypoint_idx = 0
        self.publish_path(world_path)
        self.get_logger().info(f"Path found with {len(world_path)} waypoints. Robot will start moving.")

    def publish_path(self, world_path):
        nav_msgs_path = Path()
        nav_msgs_path.header.stamp = self.get_clock().now().to_msg()
        nav_msgs_path.header.frame_id = "map"

        marker = Marker()
        marker.header = nav_msgs_path.header
        marker.ns = "path_markers"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        for (wx, wy) in world_path:
            pose = PoseStamped()
            pose.header = nav_msgs_path.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            nav_msgs_path.poses.append(pose)
            
            # Add points to marker
            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.0
            marker.points.append(p)

        self.path_pub.publish(nav_msgs_path)
        self.marker_pub.publish(marker)
        self.get_logger().info("Published path for visualization.")

    def __laser_scan_cbk(self, data):
        # Process LaserScan data to detect obstacles in front
        # Convert angle ranges to degrees for easier understanding
        angle_min = math.degrees(data.angle_min)
        angle_max = math.degrees(data.angle_max)
        angle_increment = math.degrees(data.angle_increment)
        ranges = data.ranges

        # Define the sector in front of the robot to monitor
        sector_start = -self.obstacle_angle_range / 2
        sector_end = self.obstacle_angle_range / 2

        # Calculate indices corresponding to the sector
        total_angles = len(ranges)
        angles = np.linspace(angle_min, angle_max, total_angles)
        sector_indices = np.where((angles >= sector_start) & (angles <= sector_end))[0]

        # Find the minimum distance in the sector
        valid_ranges = [ranges[i] for i in sector_indices if not math.isinf(ranges[i]) and not math.isnan(ranges[i])]
        if not valid_ranges:
            min_distance = float('inf')
        else:
            min_distance = min(valid_ranges)

        if min_distance < self.obstacle_distance_threshold:
            if not self.obstacle_detected:
                self.get_logger().warn(f"Obstacle detected within {min_distance:.2f} meters ahead. Stopping robot.")
            self.obstacle_detected = True
        else:
            if self.obstacle_detected:
                self.get_logger().info("Path is clear. Replanning path.")
                self.obstacle_detected = False
                self.replan_path()
            # Else, no change

    def replan_path(self):
        if self.goal_pose is None:
            self.get_logger().warn("No goal to replan to.")
            return
        self.plan_path()

    def control_loop(self):
        if not self.map_received or self.goal_pose is None or self.ttbot_pose is None:
            return

        if self.obstacle_detected:
            self.stop_robot()
            return

        if not self.path:
            return

        if self.current_waypoint_idx >= len(self.path):
            self.stop_robot()
            self.get_logger().info("Goal reached!")
            self.path = []
            self.goal_pose = None
            return

        target = self.path[self.current_waypoint_idx]
        dx = target[0] - self.ttbot_pose.position.x
        dy = target[1] - self.ttbot_pose.position.y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < self.goal_tolerance:
            self.current_waypoint_idx += 1
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_idx}/{len(self.path)}")
            if self.current_waypoint_idx >= len(self.path):
                self.stop_robot()
                self.get_logger().info("Final goal reached.")
                self.path = []
                self.goal_pose = None
            return

        q = self.ttbot_pose.orientation
        siny_cosp = 2.0*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        angle_to_goal = math.atan2(dy, dx)
        angle_diff = angle_to_goal - yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        cmd = Twist()
        if abs(angle_diff) > 0.3:
            cmd.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = min(dist, self.linear_speed)
            cmd.angular.z = angle_diff * 0.5
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info("Robot stopped.")

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    nav = Navigation("Navigation")
    try:
        rclpy.spin(nav)
    except KeyboardInterrupt:
        nav.get_logger().info("Navigation node terminated by user.")
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
