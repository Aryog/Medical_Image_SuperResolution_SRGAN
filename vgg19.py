from torchvision import models
import torch.nn as nn

class vgg19(nn.Module):
    def __init__(self, pre_trained=True, require_grad=False):
        super(vgg19, self).__init__()

        vgg_features = models.vgg19(pretrained=pre_trained).features
        self.seq_list = [nn.Sequential(ele) for ele in vgg_features]

        self.vgg_layer = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                          'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                          'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                          'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                          'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']

        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        vgg_outputs = []

        for layer in self.seq_list:
            x = layer(x)
            vgg_outputs.append(x)

        return vgg_outputs

# Example usage:
# Initialize the feature extractor
# -- feature_extractor = vgg19(pre_trained=True, require_grad=False)

# Assuming gen and loss_function are defined elsewhere in your code
# pred_hr = gen(lr_img)
# content_loss = loss_function(pred_hr, hr_img)

# Extract features from the high-resolution and generated images
# -- pred_features = feature_extractor(pred_hr)
# -- hr_features = feature_extractor(hr_img)
