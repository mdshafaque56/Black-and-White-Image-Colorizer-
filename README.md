# Image Colorization with OpenCV

Transform grayscale images into vibrant, colorized versions using a pre-trained deep learning model with OpenCV. This project provides a simple Python script to automate the colorization process, ideal for hobbyists, researchers, or anyone interested in image processing.

## About the Project

This tool leverages a pre-trained Caffe model to predict color information (ab channels in Lab color space) from a grayscale image's lightness (L channel). Built with OpenCV and NumPy, it processes images efficiently and is compatible with environments like Google Colab for easy experimentation.

### Built With

- **OpenCV**: For image processing and deep neural network operations.
- **NumPy**: For numerical computations and array manipulations.
- **Google Colab**: Optional, for cloud-based execution with display support.

## Getting Started

Follow these steps to set up and run the project on your local machine or in a cloud environment.

### Prerequisites

Ensure you have the following installed:

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**:
git clone https://github.com/mdshafaque56/Black-and-White-Image-Colorizer-.git
cd Black-and-White-Image-Colorizer-

2. **Install Dependencies**:
pip install -r requirements.txt
If `requirements.txt` is not yet created, install manually:
pip install opencv-python>=4.5.2 numpy>=1.21.0 google-colab

3. **Download Model Files**:
Place the pre-trained model files in the `models/` directory. Download them using the following commands or from their sources [4]:
mkdir models
wget https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy?raw=true -O ./models/pts_in_hull.npy
wget https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt -O ./models/colorization_deploy_v2.prototxt
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O ./models/colorization_release_v2.caffemodel

4. **Prepare Input Image**:
Add your grayscale image (e.g., `image1.jpg`) to the `data/raw/` directory. Create the directory if it doesn't exist:
mkdir -p data/raw

### Project Structure

| Directory/File         | Description                                      |
|------------------------|--------------------------------------------------|
| `data/raw/`            | Store input grayscale images here.              |
| `data/processed/`      | Output colorized images are saved here.         |
| `models/`              | Contains pre-trained model files.               |
| `src/colorize_image.ipynb`| Main script for image colorization.             |
| `requirements.txt`     | Lists project dependencies.                     |
| `README.md`            | Project documentation (you're reading it!).     |

## Usage

Run the script to colorize your grayscale image with a single command. Here's how:

1. Ensure your image is in `data/raw/` and model files are in `models/`.
2. Execute the script (adjust the path if necessary):
python src/colorize_image.py
3. View the result in `data/processed/colorized_output2.jpg`.

### Quick Start Code Example

Below is the core functionality of the script provided in the query. For the full script, see `src/colorize_image.py`.

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

Load model and cluster centers
prototxt = "models/colorization_deploy_v2.prototxt"
model = "models/colorization_release_v2.caffemodel"
kernel = "models/pts_in_hull.npy"
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts_in_hull = np.load(kernel)

Load grayscale image
bw_image = cv2.imread("data/raw/image1.jpg")
bw_image = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
bw_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)

Process image (simplified)
normalized = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2Lab)
l_channel = lab[:, :, 0]

... (further processing in full script)
Display and save output
cv2_imshow(colorized)
cv2.imwrite("data/processed/colorized_output2.jpg", colorized)


### Expected Results

| Step                 | Input                          | Output                          |
|----------------------|--------------------------------|---------------------------------|
| Grayscale Image      | `data/raw/image1.jpg`         | N/A                            |
| Colorization Process | Processed via Caffe model     | Lab color space conversion     |
| Final Output         | N/A                           | `data/processed/colorized_output2.jpg` |

### Understanding Lab Color Space

The project uses Lab color space for processing, which separates lightness from color information [2][3][4]:

| Channel | Description              |
|---------|--------------------------|
| L       | Lightness intensity only |
| a       | Green-Red axis          |
| b       | Blue-Yellow axis        |

This separation allows the model to use the L channel as input and predict the a and b channels for colorization.

## Features

- **Automated Colorization**: Converts grayscale to color without manual input.
- **Pre-trained Model**: Uses a robust Caffe model trained on ImageNet for accurate color prediction [4].
- **Colab Compatibility**: Includes display functions for cloud-based environments.

## Roadmap

- Add support for batch processing multiple images.
- Implement a GUI or web interface using frameworks like Streamlit for easier interaction [6].
- Optimize model inference speed for larger images.
- Allow customization of color intensity or style predictions.

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

For major changes, please open an issue first to discuss your ideas.

## License

Distributed under the MIT License. See `LICENSE` for more information if available, or contact the repository owner for licensing details.

## Contact

Have questions or suggestions? Reach out via GitHub Issues on the repository page.

## Acknowledgments

- Thanks to the OpenCV community for excellent documentation and tools.
- Inspired by research on deep learning-based image colorization by Richard Zhang et al. [3][4].
- Model files sourced from the original colorization project by Zhang et al. [4].
