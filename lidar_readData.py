#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from cv_bridge import CvBridge
import os
import csv

# Global variables
global_point_cloud = np.array([])
global_intensity = np.array([])  # Array to hold intensity values
robot_trajectory = []  # Store robot positions
robot_pose = None  # Store the latest robot pose
depth_image = None
bridge = CvBridge()  # Initialize the CvBridge
log_file_path=os.path.expanduser("~/catkin_ws/src/DRL/logfile.csv")

with open(log_file_path, 'w') as csvfile:
    pass


def log_data(event):
    global global_point_cloud
    with open(log_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point in global_point_cloud:
            writer.writerow(point)  # Log x, y, z

def lidar_callback(data):
    global global_point_cloud, global_intensity
    
    # Read points and their intensity
    gen = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z", "intensity"))
    pc_list = list(gen)
    
    # Extract XYZ coordinates and intensity into separate arrays
    pc_array = np.array([(x, y, z) for x, y, z, intensity in pc_list])
    intensity_array = np.array([intensity for x, y, z, intensity in pc_list])

    # Filter out NaN values from the intensity array and corresponding points
    valid_indices = ~np.isnan(intensity_array)
    filtered_pc_array = pc_array[valid_indices]
    filtered_intensity_array = intensity_array[valid_indices]
    
    # Apply distance-based filtering to keep points within a 40-meter range
    distances = np.linalg.norm(filtered_pc_array, axis=1)
    within_range_indices = distances <= 40
    
    global_point_cloud = filtered_pc_array[within_range_indices]
    global_intensity = filtered_intensity_array[within_range_indices]

    # Log the filtered LIDAR data
    with open(log_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point in filtered_pc_array:
            writer.writerow(point)  # Log x, y, z


def odom_callback(data):
    global robot_trajectory, robot_pose
    position = data.pose.pose.position
    robot_trajectory.append((position.x, position.y, position.z))
    robot_pose = position

def depth_callback(data):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(data, "passthrough")


def update_graph(num, ax, fig, depth_ax, intensity_ax):
    global robot_pose, global_intensity, global_point_cloud
    # Clear the previous data
    ax.clear()
    depth_ax.clear()
    intensity_ax.clear()

    # Update the point cloud plot
    if global_point_cloud.size != 0:
        ax.scatter(global_point_cloud[:, 0], global_point_cloud[:, 1], global_point_cloud[:, 2], s=1, color='blue')

        # Calculate distances in the XY plane
        distances_xy = np.linalg.norm(global_point_cloud[:, :2], axis=1)
        
        # Normalize the distances
        max_distance_xy = np.max(distances_xy)
        # Avoid division by zero
        if max_distance_xy > 0:
            normalized_distances_xy = 1 - (distances_xy / max_distance_xy)  # Closer points have higher values

            # Ensure there are no NaNs in global_intensity before multiplying
            valid_indices = ~np.isnan(global_intensity)
            valid_intensities = global_intensity
            valid_distances_xy = normalized_distances_xy[valid_indices]

            # Multiply the normalized distances with valid intensity values to emphasize closer points
            emphasized_intensity = valid_intensities * normalized_distances_xy

            # Normalize emphasized intensity if the maximum is greater than 0 to ensure it falls between 0 and 1
            max_emphasized_intensity = np.max(emphasized_intensity)
            if max_emphasized_intensity > 0:
                emphasized_intensity /= max_emphasized_intensity

            # Plot the emphasized intensity data
            point_indices = np.arange(len(emphasized_intensity))
            intensity_ax.plot(point_indices, emphasized_intensity, color='green')
            intensity_ax.set_title('LIDAR Intensity (Emphasized for Closer Points)')
            intensity_ax.set_xlabel('Point Index')
            intensity_ax.set_ylabel('Emphasized Intensity')
            intensity_ax.set_xlim(0, len(emphasized_intensity) - 1)
            intensity_ax.set_ylim(0, 1)

    # Update the depth image plot
    if depth_image is not None:
        depth_ax.imshow(depth_image, cmap='gray')
        depth_ax.set_title('Depth Image')

    plt.draw()

    # Adjust the limits based on the robot's position
    if robot_pose is not None:
        ax.set_xlim(robot_pose.x - 20, robot_pose.x + 20)
        ax.set_ylim(robot_pose.y - 20, robot_pose.y + 20)
        ax.set_zlim(robot_pose.z - 2, robot_pose.z + 18)
    else:
        set_axes_limits(ax)

    plt.draw()



    # Adjust the limits based on the robot's position
    if robot_pose is not None:
        ax.set_xlim(robot_pose.x - 20, robot_pose.x + 20)
        ax.set_ylim(robot_pose.y - 20, robot_pose.y + 20)
        ax.set_zlim(robot_pose.z - 2, robot_pose.z + 18)
    else:
        set_axes_limits(ax)

    plt.draw()


def set_axes_limits(ax):
    max_range = 10  # Set the maximum range to 40 meters
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-2, 3)  # Assuming you want to visualize from 2 meters below the sensor to the max range above
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('LIDAR Data within 40m Range')


def lidar_listener():
    global bridge  # This is to ensure we're using the global bridge variable
    rospy.init_node('lidar_listener', anonymous=True)
    
    # Subscribers
    rospy.Subscriber('velodyne_points', PointCloud2, lidar_callback)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.Subscriber('/realsense/depth/image_rect_raw', Image, depth_callback)

    rospy.Timer(rospy.Duration(60), log_data)


# Set up plotting
fig = plt.figure(figsize=(15, 5))  # Adjust the figure size as needed
ax = fig.add_subplot(131, projection='3d')  # 3D plot for the LIDAR data
depth_ax = fig.add_subplot(132)  # 2D plot for the depth image
intensity_ax = fig.add_subplot(133)  # 2D plot for the intensity

# Specify save_count in FuncAnimation
ani = FuncAnimation(fig, update_graph, fargs=(ax, fig, depth_ax, intensity_ax), interval=100, save_count=200)

lidar_listener()
plt.show()
