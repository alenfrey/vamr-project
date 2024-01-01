import cv2

# Default configurations
default_config = {
    "matcher": cv2.FlannBasedMatcher(),
    "sift": cv2.SIFT_create(),
    "ransac": {"prob": 0.99, "threshold": 1.0},
}

sift_detector = cv2.SIFT_create(
    nfeatures=2000, nOctaveLayers=5, contrastThreshold=0.05, edgeThreshold=15, sigma=1.6
)

# Specific configurations for datasets
config = {
    "parking": {
        "common": {"matcher": cv2.BFMatcher()},
        "initialization": {
            "detector": cv2.SIFT_create(
                nfeatures=2000,
                nOctaveLayers=5,
                contrastThreshold=0.05,
                edgeThreshold=15,
                sigma=1.6,
            ),
            "ransac": {"prob": 0.999, "threshold": 2.0},
        },
        "continuous": {
            "detector": cv2.SIFT_create(
                nfeatures=1000,
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.2,
            ),
            "ransac": {"prob": 0.995, "threshold": 0.5},
        },
    },
    # add other dataset configurations as needed (e.g. "kitti", "malaga", etc.)
}


# Function to get configuration
def get_config(dataset_name, mode):
    dataset_config = config.get(dataset_name, {})
    common_config = dataset_config.get("common", {})
    mode_config = dataset_config.get(mode, {})

    # Merge configurations: default < common < mode-specific
    final_config = {**default_config, **common_config, **mode_config}
    return final_config


# example usage

# from config import get_config

# # Define the dataset and mode
# dataset_name = "dataset1"
# mode = "initialization"  # This could be "initialization" or "continuous"

# # Load the configuration for the specified dataset and mode
# current_config = get_config(dataset_name, mode)

# # Extract specific configurations
# sift_detector = current_config['detector']
# matcher = current_config['matcher']
# ransac_params = current_config['ransac']

# # Now use these configurations in your VO pipeline
# # Example: Using the SIFT detector and matcher
# keypoints, descriptors = sift_detector.detectAndCompute(image, None)
# matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
# # ... rest of the VO processing

# # Example: Using RANSAC parameters
# E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=ransac_params['prob'], threshold=ransac_params['threshold'])
# # ... further processing
