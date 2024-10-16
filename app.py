from flask import Flask, jsonify, request
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import io
from PIL import Image

app = Flask(__name__)

# Global variables to store the loaded model and related objects
global_model = None
global_transform = None
global_device = None

def load_model():
    # Step 1: Define Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 2: Load the Trained Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 5)  # Assuming 5 classes
    resnet.load_state_dict(torch.load('model/fine_tuned_resnet.pth', map_location=device))
    resnet = resnet.to(device)
    resnet.eval()  # Set model to evaluation mode
    
    return resnet, transform, device

# Load the model on initialization
global_model, global_transform, global_device = load_model()

# Define class names
class_names = ['AK', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']

@app.route('/predict', methods=['GET'])
def predict():
    return jsonify({"status": "ok"})

@app.route('/inference', methods=['POST'])
def inference():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    
    # Read and process the image
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image = global_transform(image)
    image = image.unsqueeze(0)
    
    # Perform inference
    image = image.to(global_device)
    
    with torch.no_grad():
        output = global_model(image)
        _, predicted_class = torch.max(output, 1)
    
    # Get the predicted label
    predicted_label = class_names[predicted_class.item()]
    
    return jsonify({"predicted_class": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
