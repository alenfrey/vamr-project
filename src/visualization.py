import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from matplotlib import cm


class VOVisualizer:
    def __init__(self):
        self.fig = plt.figure()
        self.range = 200
        self.pose_history = []  # to store the history of poses

        self.ax_pose = self.fig.add_subplot(121, projection="3d")
        self.setup_axes()

        # subplot for the image display
        self.ax_image = self.fig.add_subplot(122)
        self.ax_image.axis("off")  # Hide axis

        plt.ion()  # turn on interactive mode
        plt.show()

    def setup_axes(self):
        self.ax_pose.set_xlabel("X axis")
        self.ax_pose.set_ylabel("Y axis")
        self.ax_pose.set_zlabel("Z axis")

    def update_plot(self, pose, image, points_3D, colors=None):
        # extract the translation vector (current position)
        t = pose[:3, 3]

        # update the pose history
        self.pose_history.append(t)

        # update the axes limits based on the current position
        self.ax_pose.clear()
        self.setup_axes()
        self.ax_pose.set_xlim([t[0] - self.range, t[0] + self.range])
        self.ax_pose.set_ylim([t[1] - self.range, t[1] + self.range])
        self.ax_pose.set_zlim([t[2] - self.range, t[2] + self.range])

        # plot the pose history
        if self.pose_history:
            history_array = np.array(self.pose_history)
            self.ax_pose.plot(
                history_array[:, 0],
                history_array[:, 1],
                history_array[:, 2],
                color="gray",
                alpha=0.5,
            )

        # plot the current pose orientation
        R = pose[:3, :3]
        self.ax_pose.quiver(
            t[0], t[1], t[2], R[0, 0], R[1, 0], R[2, 0], length=5, color="r"
        )
        self.ax_pose.quiver(
            t[0], t[1], t[2], R[0, 1], R[1, 1], R[2, 1], length=5, color="g"
        )
        self.ax_pose.quiver(
            t[0], t[1], t[2], R[0, 2], R[1, 2], R[2, 2], length=5, color="b"
        )

        if colors is not None and colors.size > 0 and points_3D is not None:
            # Ensure colors are normalized (range 0 to 1) if they are in the standard 0-255 range
            if np.max(colors) > 1:
                colors = colors / 255.0

            self.ax_pose.scatter3D(
                points_3D[0, :],
                points_3D[1, :],
                points_3D[2, :],
                c=colors,
                s=10,
            )
        elif points_3D is not None:
            self.ax_pose.scatter3D(
                points_3D[0, :],
                points_3D[1, :],
                points_3D[2, :],
                c="black",
                s=0.5,
            )

        # update the image subplot
        if image is not None:
            self.ax_image.clear()
            self.ax_image.axis("off")
            self.ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Update the plot
        plt.draw()
        plt.pause(0.00001)
