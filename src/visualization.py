import matplotlib.pyplot as plt
import numpy as np
import cv2

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class VOVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 10))

        # Subplots arrangement
        self.ax_image = self.fig.add_subplot(2, 2, 1)  # Top left
        self.ax_world = self.fig.add_subplot(2, 2, 2, projection="3d")  # Top right
        self.ax_line = self.fig.add_subplot(2, 2, 4)  # Bottom left
        self.ax_extra = self.fig.add_subplot(2, 2, 3)  # Bottom right

        self.range = 100
        self.pose_history = []  # Store the history of poses
        self.line_data = {}  # Store data for the line chart
        self.time_steps = {}  # Store time steps for each line

        self.setup_axes()
        self.adjust_layout()
        plt.ion()
        plt.show()

    def adjust_layout(self):
        # Adjust the spacing between subplots
        self.fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2
        )

    def setup_axes(self):
        # Set up the axes
        self.ax_world.set_xlabel("X")
        self.ax_world.set_ylabel("Y")
        self.ax_world.set_zlabel("Z")
        self.ax_world.set_title("3D World")

        self.ax_image.axis("off")
        self.ax_image.set_title("Current Frame")
        self.ax_line.set_title("Line Chart")
        self.ax_extra.set_title("Extra Subplot")  # Title for the extra subplot

    def update_image(self, image):
        if image is not None:
            self.ax_image.clear()
            self.ax_image.axis("off")
            self.ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def update_points_plot(self, pts_curr, pts_reprojected):
        self.ax_extra.clear()

        # Reshape pts_curr to a two-dimensional array of shape (n, 2)
        points = pts_curr.reshape(-1, 2)

        # Scatter plot of points
        x, y = points[:, 0], points[:, 1]
        self.ax_extra.scatter(x, y, c="blue", label="2D Points", alpha=0.5)

        # Get the bounding box of the image axis
        bbox = self.ax_image.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted()
        )
        width, height = bbox.width, bbox.height

        # Set the aspect of the plot to be equal, and match the image aspect ratio
        self.ax_extra.set_aspect(abs((width / height)))

        # Set limits based on the image axis size
        self.ax_extra.set_xlim(self.ax_image.get_xlim())
        self.ax_extra.set_ylim(self.ax_image.get_ylim())

        self.ax_extra.legend()
        self.ax_extra.set_title("2D Points and Reprojected Points")
        self.ax_extra.set_xlabel("x")
        self.ax_extra.set_ylabel("y")
        
        if pts_reprojected is not None:
            points = pts_reprojected.reshape(-1, 2)
            self.ax_extra.scatter(points[:, 0], points[:, 1], c="red", label="Reprojected Points", alpha=0.5)
            self.ax_extra.legend()

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

    def plot_quiver(self, pose):
        # extract the translation vector (current position)
        t = pose[:3, 3]
        R = pose[:3, :3]
        colors = ["r", "g", "b"]
        self.pose_history.append(t)
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
            )

    def update_world(
        self,
        pose,
        points_3D,
    ):
        # update the axes limits based on the current position
        self.ax_world.clear()
        self.setup_axes()
        t = pose[:3, 3]
        self.ax_world.set_xlim([t[0] - self.range, t[0] + self.range])
        self.ax_world.set_ylim([t[1] - self.range, t[1] + self.range])
        self.ax_world.set_zlim([t[2] - self.range, t[2] + self.range])
        self.plot_quiver(pose)

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

        points_3D = points_3D / 10

        self.ax_world.scatter3D(
            points_3D[0, :],
            points_3D[1, :],
            points_3D[2, :],
            c="purple",
            s=3,
        )

    def redraw(self):
        # Redraw the entire plot
        plt.draw()
        plt.pause(0.001)
