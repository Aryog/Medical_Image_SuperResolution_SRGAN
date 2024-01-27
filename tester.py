import torch
from PIL import Image
import numpy as np

from generator import Generator  # Update with correct import
# from srgan_model import Generator
def preprocess_input(image_path):
    # Load the input image
    input_image = Image.open(image_path).convert("RGB")
    input_image = np.array(input_image) / 127.5 - 1.0
    input_image = input_image.transpose(2, 0, 1).astype(np.float32)
    return torch.tensor(input_image).unsqueeze(0)

def test_single_image(generator_path, input_image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the generator model
    generator = Generator()          #for ours
    # generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16)
    generator.load_state_dict(torch.load(generator_path))
    generator = generator.to(device)
    generator.eval()

    # Preprocess the input image
    input_image = preprocess_input(input_image_path).to(device)

    # Perform inference
    with torch.no_grad():
        output = generator(input_image)
        output = output[0].cpu().numpy()
        output = (output + 1.0) / 2.0
        print(f'This is output shape before transpose: {output.shape}')
        # Add this line to check the content of the output array
        print(f'This is output content before transpose: {output}')
        output = output.transpose(1, 2, 0)
        # output = output.transpose(0, 2, 3, 1)
        result = Image.fromarray((output * 255.0).astype(np.uint8)) # for ours
        # result = Image.fromarray((output[0] * 255.0).astype(np.uint8))
        result.save(output_path)

if __name__ == "__main__":
    # Specify the paths and parameters
    # generator_path = "./model/pre_trained_model_010.pt"
    generator_path = "./model/MedSRGAN_gene_006.pt"
    input_image_path = "./check.jpeg"
    output_path = "./enhanced_output.jpeg"

    # Test the model with a single image
    test_single_image(generator_path, input_image_path, output_path)
