# MedSRGAN
## PyTorch implementation of "MedSRGAN: medical images super-resolution using generative adversarial networks"

<img src="./img/medsrgan.PNG" width="500px"></img>
<img src="./img/Discriminator.png" width="500px"></img>

```python
import torch
from generator import Generator
from discriminator import Discriminator

generator = Generator(
      in_channels= 3,
      blocks= 8
)

discriminator = Discriminator(
      in_channels= 3, 
      img_size= (256, 256)
)
```

### **Using the App**

To use the app, follow these steps:

1. Create the **`custom_dataset`** folder in your project directory.
2. Create the **`train_LR`**  and  **`train_HR`** subdirectories inside **`custom_dataset`**
3. Run the following command in the terminal to train the model:
    
    ```bash
    python main.py --LR_path custom_dataset/train_LR --GT_path custom_dataset/train_HR
    ```
    
    This will train the MedSRGAN model using your medical image dataset. Adjust the hyperparameters in the **`main.py`** file as needed.
    
4. After training, you can test the model on new images using:
    
    ```bash
    python tester.py
    ```
    
    Make sure to input the path of the test image when prompted.
5. You can check the __PSNR__ and __SSIM__ using the command
    ```bash
    python main.py --mode test --LR_path custom_dataset/train_LR --GT_path custom_dataset/train_HR --generator_path ./model/MedSRGAN_gene_0XX.pt
    ```
5. View the output result as **`enhanced_output.jpeg`** in your **root** directory.