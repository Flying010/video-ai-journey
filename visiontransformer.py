# -*- coding: utf-8 -*-
"""VisionTransformer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZefuBiQBCsxcTT-_NJMtHrW4z8tqtGEO
"""

!pip install einops

#Image patching
import torch
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

to_tensor = [Resize((144, 144)), ToTensor()]

#Compose object transforms all images into the same size and then transform into tensors
class Compose(object):
    def __init__(self, transforms):
      self.transforms = transforms

    def __call__(self, image, target):
      for t in self.transforms:
        image = t(image)
      return image, target

def show_images(images, num_samples=40, cols=8):
    """check out some of the images yesss"""
    plt.figure(figsize=(15,15))
    idx = int(len(dataset) / num_samples)
    print(images)
    for i, img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples/cols) + 1, cols, int(i/idx) + 1)
            plt.imshow(to_pil_image(img[0]))
#consists of different pets with 37 classes
dataset = OxfordIIITPet(root=".", download=True, transforms=Compose(to_tensor))
show_images(dataset)

from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor

class PatchEmbedding(nn.Module): #reshaping and then applies a linear transformation to output the embedding vectors
    def __init__(self, in_channels = 3, patch_size = 8, emb_size = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            #break down the images into patches to s1 x s2 patches and then flatten them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
# sample test
sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
print("Initial shape: ", sample_datapoint.shape)
embedding = PatchEmbedding()(sample_datapoint)
print("Patches shape: ", embedding.shape) #328 patches with a dimension of 128


'''
Input: Batch of images
Step 1: Divide each image into small patches (like splitting images into tiles)
Step 2: Flatten each patch into a vector
Step 3: Map each patch vector into an embedding space through a linear layer
Output: A sequence of patch embeddings which is then fed into the transformer encoder'''

from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout) #multi head attention already exists in pytorch

        self.q = torch.nn.Linear(dim, dim) #Linear transformation for each of these
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x): #input patch/image goes through each of them, apply attention and return attention output
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output

Attention(dim=128, n_heads=4, dropout=0.)(torch.ones((1, 5, 128))).shape

#NORMALIZATION
class PreNorm(nn.Module):
  #takes some function like Attention and applies a layernorm just before applying the function, can simply wrap ur function in this given normalization
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

norm = PreNorm(128, Attention(dim=128, n_heads=4, dropout=0.))
norm(torch.ones ((1, 5, 128))).shape

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.): #in between the linear layers, (
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), #use the Gaussian Error Linear Unit (GELU) which is the activation function used
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) #Dropout in between to avoid overfitting
      )
ff = FeedForward(dim=128, hidden_dim=256)
ff(torch.ones((1, 5, 128))).shape

#to improve the flow of information throughout training like to avoid vanishing gradients, where training signal is lost during backpropogation
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

residual_att = ResidualAdd(Attention(dim=128, n_heads=4, dropout=0.))
residual_att(torch.ones((1, 5, 128))).shape

#Full Vision Transformer Module, time to combine everything :)
from einops import repeat

class ViT(nn.Module):
    def __init__(self, ch=3, img_size=144, patch_size=4, emb_dim=32,
                n_layers=6, out_dim=37, dropout=0.1, heads=2):
    # 4 layers, small image size, patch size. etc. in order to simplify the architecture a lil to make the training faster
        super(ViT, self).__init__()

        #Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        #Patching
        #transforms input image into patch vectors, which is used in forward function
        self.patch_embedding = PatchEmbedding(in_channels=ch, patch_size=patch_size, emb_size=emb_dim)

        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim)) # dummy token for the global representation, both added to the input

        #Transformer encoders
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
          transformer_block = nn.Sequential(
              ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))), #wrapped under normalization and some residuals around this
              ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout)))) #same for feed forward
          self.layers.append(transformer_block)

        #Classification Head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))

    '''takes the classification token, which is at position 0
    uses classification token to perform predictions by passing it through another linear component, output is the output dimension like the
    number of classes'''


    def forward(self, img):
        #Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        #Add cls token to input
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) #repeat it for as many batches as we have, each batch gets one token
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] # added to each of the tokens within one batch image

        #Transformer layers
        for i in range(self.n_layers): #pass through all the transformer layers
            x = self.layers[i](x)

        #Output based on classification token
        return self.head(x[:, 0, :])

model = ViT()
print(model)
model(torch.ones((1, 3, 144, 144)))

"""## Training"""

from torch.utils.data import DataLoader
from torch.utils.data import random_split

train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=True)

import torch.optim as optim #optimizer for updating model parameters
import numpy as np #to compute mean loss easily

device = "cuda" #sets computation to GPU
model = ViT().to(device) #makes sure that the model is trained on the GPU
optimizer = optim.AdamW(model.parameters(), lr=0.001) #the AdamW optimizer is part of the training loop
criterion = nn.CrossEntropyLoss() #Loss function for classification tasks
#btw epoch is one full pass through the entire training dataset, train multiple epochs to adjust weights gradually to minimize loss
for epoch in range(1000):
    epoch_losses = [] #stores all batch losses in this epoch
    model.train()
    for step, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() #btw optimizer updates the model's parameters to minimize the loss, compute loss, then call loss.backwards to compute gradients, then adjust the parameters accordingly
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() #backpropogation
        optimizer.step()
        epoch_losses.append(loss.item()) #stores current batch loss as a scalar
    if epoch % 5 == 0:
        print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
        epoch_losses = []
        # Something was strange when using this?
        # model.eval()
        for step, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_losses.append(loss.item())
        print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)

print("Predicted classes", outputs.argmax(-1))
print("Actual classes", labels)