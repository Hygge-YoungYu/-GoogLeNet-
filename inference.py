import torch
from torchvision import transforms
from PIL import Image

# Load the trained model
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Make prediction
def predict(image_path, model):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

if __name__ == '__main__':
    model_path = 'path_to_your_model.pth'
    image_path = 'path_to_animal_image.jpg'
    model = load_model(model_path)
    probabilities = predict(image_path, model)
    print(probabilities)