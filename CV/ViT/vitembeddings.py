import torch.nn as nn
import torchvision
import cv2
import torch
image = cv2.imread('image.jpg')
image = cv2.resize(image, (224, 224))
image = torchvision.transforms.functional.to_tensor(image)
image = image.unsqueeze(0)


class PatchEmbeddings(nn.Module):
    def __init__(self):
        super(PatchEmbeddings, self).__init__()
        self.projection = nn.Conv2d(3, 768, kernel_size=16, stride=16)

    def forward(self, x):
        out = self.projection(x)
        cls = nn.AvgPool2d(kernel_size=14)(out)
        cls = cls.flatten(start_dim=2)
        out = out.flatten(start_dim=2)
        out = out.reshape((1, 196, 768))
        cls = cls.reshape((1, 1, 768))
        out = torch.cat((cls, out), 1)
        return out


class ViTEmbeddings(nn.Module):
    def __init__(self):
        super(ViTEmbeddings, self).__init__()
        self.patch_embeddings = PatchEmbeddings()
        self.drop_out = nn.Dropout(p=0.0)

    def forward(self, x):
        out = self.patch_embeddings(x)
        out = self.drop_out(out)
        return out


class ViTModel(nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.embeddings = ViTEmbeddings()

    def forward(self, x):
        out = self.embeddings(x)
        return out


model = ViTModel()
print(model(image).shape) # == ViT.from_pretrained(...)(image).shape
