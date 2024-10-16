# Grapevine Leaf Classification

This project uses a fine-tuned ResNet-18 model to classify grapevine leaf images into different varieties.

## Project Overview

This Flask application serves a machine learning model that can classify grapevine leaf images into five different varieties:
- AK
- Ala Idris
- Buzgulu
- Dimnit
- Nazli

The model is a ResNet-18 architecture fine-tuned on a custom dataset of grapevine leaf images.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the pre-trained model file `fine_tuned_resnet.pth` and place it in the project root directory.

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. The server will start running on `http://localhost:5000`.

3. To make a prediction, send a POST request to `http://localhost:5000/inference` with an image file in the request body.

   Example using curl:
   ```
   curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:5000/inference
   ```

   The response will be a JSON object with the predicted class:
   ```json
   {"predicted_class": "Buzgulu"}
   ```

## Model Details

The classification model is a ResNet-18 architecture pre-trained on ImageNet and fine-tuned on our custom dataset of grapevine leaf images. The model takes input images of size 224x224 pixels and outputs probabilities for each of the five grapevine varieties.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[Specify your license here, e.g., MIT, GPL, etc.]
