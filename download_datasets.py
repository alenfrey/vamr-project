import sys
from constants import DATA_DIR_PATH
from src.utils import download_and_unzip

# URLs for each dataset
DATASETS = {
    "parking": "https://rpg.ifi.uzh.ch/docs/teaching/2023/parking.zip",
    "kitti05": "https://rpg.ifi.uzh.ch/docs/teaching/2023/kitti05.zip",
    "malaga": "https://rpg.ifi.uzh.ch/docs/teaching/2023/malaga-urban-dataset-extract-07.zip",
}

# Lambda function for downloading a dataset
download_dataset = lambda name: download_and_unzip(DATASETS[name], DATA_DIR_PATH)


def show_menu():
    """Show the dataset menu and handle the user's choice."""
    options = ["all datasets"] + list(DATASETS.keys())
    print("\n".join(f"{i}. {option}" for i, option in enumerate(options)))
    print("\nEnter the number of the dataset you want to download.")

    try:
        choice = int(input("Your choice: "))
        if choice == 0:
            for dataset in DATASETS:
                download_dataset(dataset)
        elif 1 <= choice <= len(DATASETS):
            download_dataset(list(DATASETS.keys())[choice - 1])
        else:
            print("Invalid choice. Please enter a number from the list.")
    except ValueError:
        print("Please enter a valid number.")


if __name__ == "__main__":
    show_menu()
