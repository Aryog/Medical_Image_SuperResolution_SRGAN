import os
import torch
import torch.nn as nn
import torchvision.transforms as ttf
from PIL import Image
from generator import Generator
from featureExtractor import FeatureExtractorVGG19

# Function to load the trained generator model
def load_generator_model(epoch_number):
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(f"./gen_{epoch_number}"))
    gen.eval()
    return gen

# Function to preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Apply the same transformations as during training
    transforms = ttf.Compose([
        ttf.RandomHorizontalFlip(0.5),
        ttf.RandomVerticalFlip(0.5),
        ttf.RandomRotation((-15, 15)),
        # Add any other necessary transformations
        ttf.ToTensor()
    ])
    
    return transforms(image).unsqueeze(0).to(device)

# Function to generate the enhanced output
def generate_enhanced_output(gen, input_image):
    with torch.no_grad():
        enhanced_image = gen(input_image)
    return enhanced_image.squeeze(0).cpu()

if __name__ == "__main__":
    # Set the device (cuda or cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the path of the input image
    input_image_path = input("./")

    # Load the trained generator model (change 'epoch_number' to the desired epoch)
    epoch_number = 10  # Change this to the epoch you want to load
    generator = load_generator_model(epoch_number)

    # Preprocess the input image
    input_image = preprocess_image(input_image_path)

    # Generate the enhanced output
    enhanced_output = generate_enhanced_output(generator, input_image)

    # Convert the enhanced output to a NumPy array for visualization
    enhanced_output_np = enhanced_output.numpy()

    # Save or display the enhanced output as needed
    enhanced_output_pil = ttf.ToPILImage()(enhanced_output.cpu())
    enhanced_output_pil.show()
