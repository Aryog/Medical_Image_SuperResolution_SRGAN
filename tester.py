import os
import torch
import torchvision.transforms as ttf
from PIL import Image
from generator import Generator
from featureExtractor import FeatureExtractorVGG19
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load the trained generator model
def load_generator_model(epoch_number, model_dir="./genModel/"):
    # Load the generator model
    gen = Generator(n_channels=3, blocks=8)
    gen.load_state_dict(torch.load(os.path.join(model_dir, f"gen_{epoch_number}")))
    gen = gen.to(device)
    gen.eval()
    return gen

# Function to preprocess the input image
def preprocess_image(input_image_path, transforms):
    image = Image.open(input_image_path).convert('RGB')
    
    # Apply the same transformations as during training
    image = transforms(image)
    print("preprocess_image")
    print(image)
    
    # Normalize the image to be in the range [-1, 1] (matching GAN_Data normalization)
    image = (image / 127.5) - 1.0
    
    return image.unsqueeze(0).to(device)

# Function to generate the enhanced output
def generate_enhanced_output(gen, input_image):
    with torch.no_grad():
        enhanced_image = gen(input_image)
    return enhanced_image.squeeze(0).cpu()

if __name__ == "__main__":
    # Set the device (cuda or cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the path of the input image
    input_image_path = input("Enter the path of the input image: ")

    # Load the trained generator model (change 'epoch_number' to the desired epoch)
    epoch_number = 10  # Change this to the epoch you want to load
    generator = load_generator_model(epoch_number-1)

    # Define the transformations for preprocessing (matching GAN_Data)
    transforms = ttf.Compose([
        ttf.Resize((256, 256)),
        ttf.GaussianBlur(3, sigma=(0.1, 2.0)),
        ttf.ToTensor()
    ])

    # Preprocess the input image
    input_image = preprocess_image(input_image_path, transforms)
    # Print input image tensor
    print("Input Image Tensor:")
    print(input_image)

    # Generate the enhanced output
    enhanced_output = generate_enhanced_output(generator, input_image)

    # Normalize the enhanced output tensor to be in the range [0, 1]
    enhanced_output = torch.clamp(enhanced_output, 0, 1)

    # Convert the enhanced output to a NumPy array for visualization
    enhanced_output_np = enhanced_output.mul(255).byte().numpy()

    # Save the enhanced output as an image
    output_image_path = "./enhanced_output.png"
    enhanced_output_pil = Image.fromarray(enhanced_output_np.transpose(1, 2, 0).astype(np.uint8))
    enhanced_output_pil.save(output_image_path)