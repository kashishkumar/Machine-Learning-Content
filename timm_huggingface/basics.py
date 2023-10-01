import torch
import timm 

from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)

# List efficientnet models for pytorch
model_names_efficientnet = timm.list_models('*efficientnet*', pretrained=True)
pprint(model_names_efficientnet)

# Finetune a pretrained model
model_mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)



# Feature Extraction
model = timm.create_model('vit_base_patch16_224', pretrained=True)
print(model)

output = model(torch.randn(1, 3, 224, 224))
print(output.shape)  # torch.Size([1, 1000])

model.forward_features(torch.randn(1, 3, 224, 224)).shape

# EfficientNet
model_efficientnet = timm.create_model('efficientnet_b3', pretrained=True)
output = model_efficientnet(torch.randn(1, 3, 300, 300))
print(output.shape)  # torch.Size([1, 1000])

model_efficientnet.forward_features(torch.randn(1, 3, 300, 300)).shape

