import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class RealTimePoseVisualizer:
    def __init__(self):
        # Initialize the plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim([0, 30])
        self.ax.set_ylim([0, 30])
        self.ax.set_zlim([0, 30])
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.set_zlabel("Z axis")

        plt.ion()  # Turn on interactive mode
        plt.show()

    def update_plot(self, pose, landmarks):
        # Clear the plot
        self.ax.clear()
        self.ax.set_xlim([0, 30])
        self.ax.set_ylim([0, 30])
        self.ax.set_zlim([0, 30])
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.set_zlabel("Z axis")

        # Extract the rotation matrix and translation vector
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Plot the car's orientation
        self.ax.quiver(
            t[0], t[1], t[2], R[0, 0], R[1, 0], R[2, 0], length=3, color="r"
        )
        self.ax.quiver(
            t[0], t[1], t[2], R[0, 1], R[1, 1], R[2, 1], length=3, color="g"
        )
        self.ax.quiver(
            t[0], t[1], t[2], R[0, 2], R[1, 2], R[2, 2], length=3, color="b"
        )

        # Plot the landmarks
        # self.ax.scatter3D(
        #     landmarks[0, :],
        #     landmarks[1, :],
        #     landmarks[2, :],
        #     c="black",
        #     s=0.5,
        # )
        
        # self.ax.scatter(
        #     keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c="blue", marker="o"
        # )

        # Update the plot
        plt.draw()
        plt.pause(1)


# Usage Example
# visualizer = RealTimePoseVisualizer()

# # Example data (replace with your real data in the main loop)
# pose = np.array([[1, 0, 0, 54.45], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# keypoints = np.random.rand(20, 3) * 100

# # Update the plot (call this in your main loop)
# visualizer.update_plot(pose, keypoints)
