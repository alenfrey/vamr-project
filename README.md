# Vision Algorithms for Mobile Robotics - Project

## Screencasts

* Parking: https://youtu.be/W_FU94QPROM
* Kitti: https://youtu.be/BZOVhOeJ-Cw
* Malaga: https://youtu.be/CdZAbssPqkc

The screencasts were recorded on a Macbook Pro M1 with 16 GiB. 
The code was executed in the conda environment on the CPU.

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (for managing the environment)
- [Python 3.9](https://www.python.org/downloads/)

### Setting Up the Conda Environment
1. Clone the repository to your local machine:
   ```
   git clone https://github.com/alenfrey/vamr-project
   cd vamr-project
   ```

2. Create and activate the Conda environment, and install the required packages:
   ```
   conda env create -f env.yml
   conda activate vamr
   ```

## Usage

### Downloading the Datasets
To download the datasets, run the python script `download_datasets.py` from the project root directory:
   ```
   python3 download_datasets.py
   ```

## Testing

This project uses `pytest` and `Hypothesis` for testing to ensure code quality and reliability. Here's how you can run tests and contribute to them.

### Running Tests

To run the tests, navigate to the project root directory in your terminal and execute the following command:

```bash
pytest
```

`pytest` will automatically discover and run all tests in the project, including those generated by `Hypothesis`.

### Writing Tests

Our tests are located in the `tests` directory. When writing tests, its helpful to adhere to the following guidelines:

#### Naming Conventions
   - Test files should be named `test_<module>.py` where `<module>` corresponds to the module being tested.
   - Test functions should start with `test_`.

#### Structure
   - Group tests logically by functionality within the test files.
   - Use `Hypothesis` for generating a wide range of input scenarios, especially for edge cases.


## Formatting Code
Run the following command from the root directory of the project with the Conda environment activated to format the code using Black:
   ```
   black .
   ```
   
This is important for being able to compare code changes in git (git diff) in a meaningful way without being distracted by formatting differences.

## Docstring Style
ToDo: Discuss, decide and add docstring style 

