import torch.nn as nn
import torch
import torchvision.models as models
# import torchvision.transforms as transforms
# from dataset import GAN_Data

class RWMAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print('In RWMAB')
        x_ = self.layer1(x)
        x__ = self.layer2(x_)

        x = x__ * x_ + x

        return x


class ShortResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layers = nn.ModuleList([RWMAB(in_channels) for _ in range(16)])

    def forward(self, x):
        # print('In Short Residual Block')
        x_ = x.clone()

        for layer in self.layers:
            x_ = layer(x_)

        return x_ + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, blocks=8):
        super().__init__()

        # Adding the noise part here

        self.conv = nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1)

        self.short_blocks = nn.ModuleList(
            [ShortResidualBlock(64) for _ in range(blocks)]
        )

        self.conv2 = nn.Conv2d(64, 64, (1, 1), stride=1, padding=0)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),  # Remove if output is 2x the input
            nn.Conv2d(64, 3, (1, 1), stride=1, padding=0),  # Change 64 -> 256
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print('In Generator')
        x = self.conv(x)
        x_ = x.clone()

        for layer in self.short_blocks:
            x_ = layer(x_)

        x = torch.cat([self.conv2(x_), x], dim=1)

        x = self.conv3(x)

        return x



# if __name__ == "__main__":
#     # Step 1: Create an instance of the Generator
#     generator = Generator(in_channels=3, blocks=8)

#     # Step 2: Prepare the dataset
#     path_list = [...]  # List of paths to your images
#     dataset = GAN_Data(path_list, transforms=transforms.ToTensor())  # Assuming ToTensor() is sufficient

#     # Step 3: Create a DataLoader for training
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#     # Step 4: Define loss function and optimizer
#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

#     # Step 5: Training loop
#     epochs = 10

#     for epoch in range(epochs):
#         for batch in dataloader:
#             lr_img, hr_img = batch
#             fake_img = generator(lr_img)

#             loss = criterion(fake_img, hr_img)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

#     # Optional: Save the trained model
#     torch.save(generator.state_dict(), 'generator_model.pth')