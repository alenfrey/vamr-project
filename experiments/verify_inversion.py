import numpy as np

# Define a sample rotation matrix R and translation vector t
R = np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]])
t = np.array([[1], [2], [3]])

# Create the initial transformation matrix
transformation_matrix = np.hstack((R, t))
transformation_matrix = np.vstack([transformation_matrix, [0, 0, 0, 1]])

# Manually invert the transformation matrix as per the given method
R_0_1 = np.linalg.inv(R)
relative_translation = -R_0_1.dot(t)
manual_inverse_transformation = np.hstack((R_0_1, relative_translation))
manual_inverse_transformation = np.vstack([manual_inverse_transformation, [0, 0, 0, 1]])

# Invert the transformation matrix using np.linalg.inv
inverse_transformation = np.linalg.inv(transformation_matrix)

# Compare the results
comparison = np.isclose(manual_inverse_transformation, inverse_transformation)

print("Manual Inverse Transformation Matrix:\n", manual_inverse_transformation)
print("Inverse Transformation Matrix:\n", inverse_transformation)
print("Comparison:\n", comparison)
