#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import time
import heapq
import numpy as np
from collections import deque
from math import ceil

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Twist, Point, PoseStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('task1_algorithm')

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.frontier_marker_pub = self.create_publisher(Marker, 'frontier_markers', 10)
        self.path_pub = self.create_publisher(Path, 'path', 10)

        # Internal state
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.05
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

        self.inflated_map = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.laser_ranges = None
        self.laser_angle_min = 0.0
        self.laser_angle_inc = 0.0

        self.last_frontier_search_time = 0.0
        self.frontier_search_interval = 10.0
        self.exploration_start_time = time.time()
        self.max_exploration_time = 1800.0

        self.current_path = []
        self.current_goal = None
        self.visited_frontiers = []

        self.timer = self.create_timer(0.5, self.exploration_loop)

        # Robot and planning parameters
        self.robot_radius = 0.2
        self.safety_margin = 1.2
        self.linear_speed = 0.15
        self.angular_speed = 0.2
        self.num_smoothing_passes = 3
        self.collision_distance_threshold = 0.25

        self.inflation_radius = 2 

        # Wall bouncing mode (fallback mechanism)
        self.wall_bouncing_mode = False
        self.wall_bounce_turning = False
        self.wall_bounce_turn_start = 0.0
        self.wall_bounce_turn_duration = 2.0

    def map_callback(self, msg: OccupancyGrid):
        data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_data = data
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        needed_cells = ceil((self.robot_radius * self.safety_margin) / self.map_resolution)
        self.inflation_radius = max(needed_cells, 1)

        self.inflated_map = self.inflate_obstacles(data)

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0*(q.w*q.z+q.x*q.y)
        cosy_cosp = 1.0-2.0*(q.y*q.y+q.z*q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg: LaserScan):
        self.laser_ranges = msg.ranges
        self.laser_angle_min = msg.angle_min
        self.laser_angle_inc = msg.angle_increment

    def exploration_loop(self):
        if (time.time() - self.exploration_start_time) > self.max_exploration_time:
            self.get_logger().info("Max exploration time reached. Stopping exploration.")
            self.stop_robot()
            return

        # If we have a path and goal, follow it
        if self.current_path and self.current_goal and not self.wall_bouncing_mode:
            if self.check_collision_imminent():
                self.get_logger().warn("Collision imminent! Stopping and switching to wall bouncing mode.")
                self.stop_robot()
                self.current_path = []
                self.current_goal = None
                self.wall_bouncing_mode = True
            else:
                if self.follow_path():
                    # If path finished, append current_goal if not None
                    # to visited_frontiers
                    if self.current_goal is not None:
                        self.visited_frontiers.append(self.current_goal)
                    self.current_path = []
                    self.current_goal = None
            return

        if self.map_data is None or self.inflated_map is None:
            # Map not ready
            return

        # If in wall bouncing mode, try to exit it by finding a new path
        if self.wall_bouncing_mode:
            frontiers = self.find_frontiers_global_map()
            frontiers = [f for f in frontiers if self.is_new_frontier(f)]
            if frontiers:
                clusters = self.cluster_frontiers(frontiers)
                best_score = -1.0
                best_cluster = None
                for cluster in clusters:
                    centroid = self.cluster_centroid(cluster)
                    cluster_size = len(cluster)
                    dist = self.dist_to_robot(centroid)
                    score = cluster_size / (dist+1.0)
                    if score > best_score:
                        best_score = score
                        best_cluster = cluster
                frontier_goal = self.cluster_centroid(best_cluster)
                path = self.plan_path(frontier_goal)
                if path:
                    for _ in range(self.num_smoothing_passes):
                        path = self.smooth_path(path)
                    self.current_goal = frontier_goal
                    self.current_path = path
                    self.publish_path(path)
                    self.wall_bouncing_mode = False
                    return
            # Continue wall bouncing if no path found
            self.wall_bouncing_behavior()
            return

        # Normal operation: find new frontiers periodically
        if (time.time() - self.last_frontier_search_time) > self.frontier_search_interval:
            self.last_frontier_search_time = time.time()
            frontiers = self.find_frontiers_global_map()
            frontiers = [f for f in frontiers if self.is_new_frontier(f)]
            if not frontiers:
                self.get_logger().info("No new frontiers found. Exploration complete.")
                self.stop_robot()
                return

            self.publish_frontier_markers(frontiers)
            clusters = self.cluster_frontiers(frontiers)
            best_score = -1.0
            best_cluster = None
            for cluster in clusters:
                centroid = self.cluster_centroid(cluster)
                cluster_size = len(cluster)
                dist = self.dist_to_robot(centroid)
                score = cluster_size / (dist+1.0)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            frontier_goal = self.cluster_centroid(best_cluster)
            path = self.plan_path(frontier_goal)
            if path:
                for _ in range(self.num_smoothing_passes):
                    path = self.smooth_path(path)
                self.current_goal = frontier_goal
                self.current_path = path
                self.publish_path(path)
            else:
                # No path found, switch to wall bouncing mode
                self.get_logger().warn("No path found to chosen cluster. Switching to wall bouncing mode.")
                self.wall_bouncing_mode = True

    def wall_bouncing_behavior(self):
        if self.wall_bounce_turning:
            if time.time() - self.wall_bounce_turn_start < self.wall_bounce_turn_duration:
                # Keep turning left
                cmd = Twist()
                cmd.angular.z = self.angular_speed
                self.cmd_vel_pub.publish(cmd)
            else:
                self.wall_bounce_turning = False
        else:
            # Move forward if clear, else turn
            if self.check_collision_imminent():
                self.wall_bounce_turning = True
                self.wall_bounce_turn_start = time.time()
                cmd = Twist()
                cmd.angular.z = self.angular_speed
                self.cmd_vel_pub.publish(cmd)
            else:
                cmd = Twist()
                cmd.linear.x = self.linear_speed
                self.cmd_vel_pub.publish(cmd)

    def is_new_frontier(self, frontier):
        # Filter out None values in visited_frontiers
        valid_visited = [vf for vf in self.visited_frontiers if vf is not None]
        for vf in valid_visited:
            if self.euclidean_distance(frontier, vf) < 0.3:
                return False
        if self.dist_to_robot(frontier) < 0.1:
            return False
        return True

    def find_frontiers_global_map(self):
        frontiers = []
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self.map_data[y, x] == 0:
                    neighbors = [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
                    for (ny, nx) in neighbors:
                        if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                            if self.map_data[ny, nx] == -1:
                                wx = x*self.map_resolution+self.map_origin_x
                                wy = y*self.map_resolution+self.map_origin_y
                                frontiers.append((wy, wx))
                                break
        return frontiers

    def cluster_frontiers(self, frontiers):
        visited = set()
        clusters = []
        for i, f in enumerate(frontiers):
            if i in visited:
                continue
            queue = deque([i])
            cluster = []
            while queue:
                idx = queue.popleft()
                if idx in visited:
                    continue
                visited.add(idx)
                cluster.append(frontiers[idx])
                for j, f2 in enumerate(frontiers):
                    if j not in visited:
                        if self.euclidean_distance(frontiers[idx], f2) < 0.5:
                            queue.append(j)
            clusters.append(cluster)
        return clusters

    def cluster_centroid(self, cluster):
        cx = np.mean([c[1] for c in cluster])
        cy = np.mean([c[0] for c in cluster])
        return (cy, cx)

    def euclidean_distance(self, f1, f2):
        return math.sqrt((f1[0]-f2[0])**2+(f1[1]-f2[1])**2)

    def dist_to_robot(self, f):
        return math.sqrt((f[1]-self.robot_x)**2+(f[0]-self.robot_y)**2)

    def inflate_obstacles(self, data):
        inflated = np.copy(data)
        mask = (data > 50)
        for y in range(self.map_height):
            for x in range(self.map_width):
                if mask[y, x]:
                    for iy in range(y-self.inflation_radius, y+self.inflation_radius+1):
                        for ix in range(x-self.inflation_radius, x+self.inflation_radius+1):
                            if 0 <= iy < self.map_height and 0 <= ix < self.map_width:
                                inflated[iy, ix] = 100
        return inflated

    def plan_path(self, goal):
        gx = goal[1]
        gy = goal[0]
        start_idx = self.world_to_index(self.robot_x, self.robot_y)
        goal_idx = self.world_to_index(gx, gy)
        if start_idx is None or goal_idx is None:
            return []
        if not self.is_free_cell(goal_idx):
            return []

        open_list = []
        heapq.heappush(open_list, (0, start_idx))
        came_from = {start_idx: None}
        g_score = {start_idx: 0}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal_idx:
                return self.reconstruct_path(came_from, current)

            for neigh in self.get_neighbors(current):
                if not self.is_free_cell(neigh):
                    continue
                tentative_g = g_score[current] + 1
                if neigh not in g_score or tentative_g < g_score[neigh]:
                    g_score[neigh] = tentative_g
                    f_score = tentative_g + self.heuristic(neigh, goal_idx)
                    came_from[neigh] = current
                    heapq.heappush(open_list, (f_score, neigh))
        return []

    def heuristic(self, idx1, idx2):
        x1, y1 = self.index_to_xy(idx1)
        x2, y2 = self.index_to_xy(idx2)
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def get_neighbors(self, idx):
        x = idx % self.map_width
        y = idx // self.map_width
        nbrs = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx = x+dx
            ny = y+dy
            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                nbrs.append(ny*self.map_width+nx)
        return nbrs

    def is_free_cell(self, idx):
        return self.inflated_map[idx//self.map_width, idx%self.map_width] < 50

    def world_to_index(self, wx, wy):
        mx = int((wx - self.map_origin_x)/self.map_resolution)
        my = int((wy - self.map_origin_y)/self.map_resolution)
        if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
            return None
        return my*self.map_width+mx

    def index_to_xy(self, idx):
        x = idx % self.map_width
        y = idx // self.map_width
        return x, y

    def reconstruct_path(self, came_from, current):
        path = []
        while current is not None:
            x = current % self.map_width
            y = current // self.map_width
            wx = x*self.map_resolution + self.map_origin_x
            wy = y*self.map_resolution + self.map_origin_y
            path.append((wx, wy))
            current = came_from[current]
        path.reverse()
        return path

    def smooth_path(self, path):
        if len(path) < 3:
            return path
        changed = True
        while changed:
            changed = False
            new_path = [path[0]]
            i = 0
            while i < len(path)-1:
                j = len(path)-1
                found = False
                while j > i+1:
                    if self.check_line_of_sight(new_path[-1], path[j]):
                        new_path.append(path[j])
                        i = j
                        found = True
                        break
                    j -= 1
                if not found:
                    if i < len(path)-1:
                        new_path.append(path[i+1])
                    i += 1
            if len(new_path) < len(path):
                changed = True
                path = new_path
        return path

    def check_line_of_sight(self, p1, p2):
        cells = self.raytrace(p1, p2)
        for c in cells:
            if not self.is_free_cell_index(c):
                return False
        return True

    def raytrace(self, p1, p2):
        x0i, y0i = self.world_to_map_idx(p1[0], p1[1])
        x1i, y1i = self.world_to_map_idx(p2[0], p2[1])
        if x0i is None or x1i is None:
            return []
        return self.bresenham_line(x0i, y0i, x1i, y1i)

    def world_to_map_idx(self, wx, wy):
        mx = int((wx - self.map_origin_x)/self.map_resolution)
        my = int((wy - self.map_origin_y)/self.map_resolution)
        if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
            return None, None
        return mx, my

    def bresenham_line(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx+dy
        x, y = x0, y0
        while True:
            cells.append((x,y))
            if x == x1 and y == y1:
                break
            e2 = 2*err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return cells

    def is_free_cell_index(self, cell):
        x,y = cell
        if x < 0 or y < 0 or x >= self.map_width or y >= self.map_height:
            return False
        return self.inflated_map[y,x] < 50

    def follow_path(self):
        if not self.current_path:
            return True
        target = self.current_path[0]
        dist = math.sqrt((target[0]-self.robot_x)**2+(target[1]-self.robot_y)**2)
        if dist < 0.05:
            self.current_path.pop(0)
            if not self.current_path:
                self.stop_robot()
                if self.current_goal is not None:
                    self.visited_frontiers.append(self.current_goal)
                self.current_goal = None
                return True
            return False

        angle_to_goal = math.atan2(target[1]-self.robot_y, target[0]-self.robot_x)
        angle_diff = angle_to_goal - self.robot_yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        cmd = Twist()
        if abs(angle_diff) > 0.2:
            cmd.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = angle_diff * 0.5
        self.cmd_vel_pub.publish(cmd)
        return False

    def check_collision_imminent(self):
        if self.laser_ranges is None:
            return False
        forward_angle = 0.0
        angle_range = 10.0*math.pi/180.0
        min_distance = float('inf')

        start_angle = forward_angle - angle_range
        end_angle = forward_angle + angle_range

        start_idx = int((start_angle - self.laser_angle_min)/self.laser_angle_inc)
        end_idx = int((end_angle - self.laser_angle_min)/self.laser_angle_inc)

        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(self.laser_ranges)-1)

        for i in range(start_idx, end_idx+1):
            if 0 <= i < len(self.laser_ranges):
                r = self.laser_ranges[i]
                if r < min_distance:
                    min_distance = r

        if min_distance < self.collision_distance_threshold:
            return True

        return False

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def publish_frontier_markers(self, frontiers):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "frontiers"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for f in frontiers:
            p = Point()
            p.x = f[1]
            p.y = f[0]
            p.z = 0.0
            marker.points.append(p)

        self.frontier_marker_pub.publish(marker)

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for (wx, wy) in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
