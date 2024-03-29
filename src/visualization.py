import matplotlib.pyplot as plt
import numpy as np
import cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import cm
from src.utils import timer
from collections import deque



class VOsualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 10))
        self.connect_events()
        gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

        # Top row - spanning full width
        self.ax_image = self.fig.add_subplot(gs[0, 0:2])  # Top left (full width)
        self.ax_world = self.fig.add_subplot(
            gs[0, 2], projection="3d"
        )  # Top right (3D plot)

        # Bottom row - three plots
        self.ax_points = self.fig.add_subplot(gs[1, 0])  # Bottom left
        self.ax_line = self.fig.add_subplot(gs[1, 1])  # Bottom middle
        self.ax_global_traj = self.fig.add_subplot(
            gs[1, 2], projection="3d"
        )  # Bottom right (3D plot)

        self.range = 20
        self.pose_history = []  # Store the history of poses
        self.ground_truth_pose_history = []  # Store the history of ground truth poses
        self.line_data = {}  # Store data for the line chart
        self.time_steps = {}  # Store time steps for each line

        self.all_points = set()  # set to hold all world points for mapping

        self.max_points_length = 10000  # Adjust this number based on your needs
        self.all_points_deque = deque(maxlen=self.max_points_length)
        self.all_points_set = set()

        self.setup_axes()
        self.adjust_layout()
        plt.ion()
        plt.show()

    def connect_events(self):
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event):
        # TODO: set interactive
        if event.key == "up":
            pass
        elif event.key == "down":
            pass

    def adjust_layout(self):
        # Adjust the spacing between subplots
        self.fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1
        )

    def setup_axes(self):
        # Set up the axes
        self.ax_world.set_xlabel("X")
        self.ax_world.set_ylabel("Y")
        self.ax_world.set_zlabel("Z")
        self.ax_world.set_title("3D World")
        
        self.ax_global_traj.set_xlabel("X")
        self.ax_global_traj.set_ylabel("Y")
        self.ax_global_traj.set_zlabel("Z")
        self.ax_global_traj.set_title("Global Trajectory")

        self.ax_global_traj.set_xlabel("X")
        self.ax_global_traj.set_ylabel("Y")
        self.ax_global_traj.set_zlabel("Z")
        self.ax_global_traj.set_title("Global Trajectory")

        self.ax_image.axis("off")
        self.ax_image.set_title("Current Frame")
        self.ax_line.set_title("Line Chart")
        self.ax_points.set_title("Reprojection")
        # self.ax_extra.set_title("Extra Subplot")  # Title for the extra subplot

    def update_image(self, image):
        if image is not None:
            self.ax_image.clear()
            self.ax_image.axis("off")
            self.ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def update_points_plot(self, pts_curr, pts_reprojected):
        self.ax_points.clear()

        # Reshape pts_curr to a two-dimensional array of shape (n, 2)
        points = pts_curr.reshape(-1, 2)

        # Scatter plot of points
        x, y = points[:, 0], points[:, 1]
        self.ax_points.scatter(
            x, y, edgecolors="black", facecolors="white", label="2D Points", s=60
        )

        # Get the bounding box of the image axis
        bbox = self.ax_image.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted()
        )
        width, height = bbox.width, bbox.height

        # Set the aspect of the plot to be equal, and match the image aspect ratio
        self.ax_points.set_aspect(abs((width / height)))

        # Set limits based on the image axis size
        self.ax_points.set_xlim(self.ax_image.get_xlim())
        self.ax_points.set_ylim(self.ax_image.get_ylim())

        self.ax_points.legend()
        self.ax_points.set_title("2D Points and Reprojected Points")
        self.ax_points.set_xlabel("x")
        self.ax_points.set_ylabel("y")

        if pts_reprojected is not None:
            points = pts_reprojected.reshape(-1, 2)
            self.ax_points.scatter(
                points[:, 0],
                points[:, 1],
                c="blue",
                label="Reprojected Points",
                alpha=0.7,
                s=10,
            )
            self.ax_points.legend()

    def update_line_chart(self, new_data):
        if new_data is None:
            return
        """
        Update the line chart with new data.

        new_data: dict
            A dictionary where keys are the labels for each data series and
            values are tuples (new_value, time_step).
        """
        for label, (value, time_step) in new_data.items():
            if label not in self.line_data:
                self.line_data[label] = []
                self.time_steps[label] = []

            self.line_data[label].append(value)
            self.time_steps[label].append(time_step)

        self.ax_line.clear()
        for label, data in self.line_data.items():
            self.ax_line.plot(self.time_steps[label], data, label=label)

        self.ax_line.legend()
        self.ax_line.relim()
        self.ax_line.autoscale_view()
        self.ax_line.set_title("Line Chart")

    def plot_quiver(self, pose, alpha=1.0):
        # extract the translation vector (current position)
        t = pose[:3, 3]
        R = pose[:3, :3]
        colors = ["r", "g", "b"]
        # plotting each axis component
        for i in range(3):
            self.ax_world.quiver(
                t[0],
                t[1],
                t[2],
                R[0, i],
                R[1, i],
                R[2, i],
                length=self.range // 8,
                color=colors[i],
                alpha=alpha,
            )

    @timer
    def update_world(
        self,
        pose,
        points_3D,
        ground_truth_pose=None,
        colors=None,
    ):
        # update the axes limits based on the current position
        self.ax_world.clear()
        self.setup_axes()
        t = pose[:3, 3]
        self.ax_world.set_xlim([t[0] - self.range, t[0] + self.range])
        self.ax_world.set_ylim([t[1] - self.range, t[1] + self.range])
        self.ax_world.set_zlim([t[2] - self.range, t[2] + self.range])
        self.plot_quiver(pose)

        self.pose_history.append(t)
        for point in points_3D.T:
            point_tuple = tuple(point)
            if point_tuple not in self.all_points_set:
                self.all_points_set.add(point_tuple)
                self.all_points_deque.append(point_tuple)

                # If deque is full, remove the oldest point from the set
                if len(self.all_points_deque) == self.max_points_length:
                    oldest_point = self.all_points_deque[0]
                    self.all_points_set.remove(oldest_point)

        if ground_truth_pose is not None:
            # self.plot_quiver(ground_truth_pose, alpha=0.5)
            self.ground_truth_pose_history.append(ground_truth_pose[:3, 3])
            history_array = np.array(self.ground_truth_pose_history)
            self.ax_world.plot(
                history_array[:, 0],
                history_array[:, 1],
                history_array[:, 2],
                color="green",
                alpha=0.5,
            )

        # plot the pose history
        if self.pose_history:
            history_array = np.array(self.pose_history)
            self.ax_world.plot(
                history_array[:, 0],
                history_array[:, 1],
                history_array[:, 2],
                color="gray",
                alpha=0.5,
            )

        # scale points_3D
        if points_3D is None:
            return

        # Plot all points if available
        if self.all_points_deque:
            all_points_array = np.array(
                list(self.all_points_deque)
            ).T  # Convert to 3xN shape
            self.ax_world.scatter3D(
                all_points_array[0, :],
                all_points_array[1, :],
                all_points_array[2, :],
                c="gray",
                s=0.5,
                alpha=0.1,
                zorder=0,
            )

        self.ax_world.scatter3D(
            points_3D[0, :],
            points_3D[1, :],
            points_3D[2, :],
            c=colors if colors is not None else "blue",
            s=3,
            zorder=1,
        )

    def update_global_view(self):
        # Clear the axis for fresh plotting
        self.ax_global_traj.clear()

        # Set up the axis labels and title
        self.ax_global_traj.set_xlabel("X")
        self.ax_global_traj.set_ylabel("Y")
        self.ax_global_traj.set_zlabel("Z")
        self.ax_global_traj.set_title("Global Trajectory")

        # Define a large range for a zoomed-out view
        axis_range = 100  # You can adjust this value as needed

        # Plot the trajectory history if available
        if self.pose_history:
            history_array = np.array(self.pose_history)

            # Determine the bounds of the trajectory
            min_bounds = history_array.min(axis=0) - axis_range
            max_bounds = history_array.max(axis=0) + axis_range

            # Set axis limits
            self.ax_global_traj.set_xlim(min_bounds[0], max_bounds[0])
            self.ax_global_traj.set_ylim(min_bounds[1], max_bounds[1])
            self.ax_global_traj.set_zlim(min_bounds[2], max_bounds[2])

            self.ax_global_traj.plot(
                history_array[:, 0],
                history_array[:, 1],
                history_array[:, 2],
                color="blue",
                alpha=0.7,
                linewidth=2)

    def redraw(self):
        # Redraw the entire plot
        self.update_global_view()
        plt.draw()
        plt.pause(0.001)


def scatter_3d_points(points_3d, colors=None, title="3D World"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # visualize depth of points
    scatter = ax.scatter(
        points_3d[0],
        points_3d[1],
        points_3d[2],
        c=colors,
        marker="o",
        s=12,
        cmap="gray",
        alpha=1.0,
    )

    ax.view_init(elev=-90, azim=-90)  # viewpoint
    ax.scatter(0.3, 0, 0, c="red", marker="o", s=300)  # pose
    plt.show()
