import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


# Set random seed for reproducibility
torch.manual_seed(111)


device = torch.device("cpu")  # Force using CPU


# Transformations
transform = transforms.Compose(
   [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


# Download FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
   root=".", train=True, download=True, transform=transform
)


# Create DataLoader
batch_size = 16  # Reduced batch size for faster training
train_loader = torch.utils.data.DataLoader(
   train_set, batch_size=batch_size, shuffle=True, num_workers=2
)


# Plot some sample images from the dataset
real_samples, fashion_labels = next(iter(train_loader))
for i in range(16):
   ax = plt.subplot(4, 4, i + 1)
   plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
   plt.xticks([])
   plt.yticks([])


# Define the Discriminator model
class Discriminator(nn.Module):
   def __init__(self):
       super().__init__()
       self.model = nn.Sequential(
           nn.Linear(784, 512),  # Reduced layer size
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(512, 256),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(256, 1),
           nn.Sigmoid(),
       )


   def forward(self, x):
       x = x.view(x.size(0), 784)
       output = self.model(x)
       return output


# Initialize Discriminator
discriminator = Discriminator().to(device=device)


# Define the Generator model
class Generator(nn.Module):
   def __init__(self):
       super().__init__()
       self.model = nn.Sequential(
           nn.Linear(100, 128),  # Reduced layer size
           nn.ReLU(),
           nn.Linear(128, 256),
           nn.ReLU(),
           nn.Linear(256, 512),
           nn.ReLU(),
           nn.Linear(512, 784),
           nn.Tanh(),
       )


   def forward(self, x):
       output = self.model(x)
       output = output.view(x.size(0), 1, 28, 28)
       return output


# Initialize Generator
generator = Generator().to(device=device)


# Hyperparameters
lr = 0.0002  # Adjusted learning rate
num_epochs = 10  # Reduced number of epochs for faster training
loss_function = nn.BCELoss()


# Optimizers
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)


# Training loop
for epoch in range(num_epochs):
   for n, (real_samples, fashion_labels) in enumerate(train_loader):
       # Data for training the discriminator
       real_samples = real_samples.to(device=device)
       real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
       latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
       generated_samples = generator(latent_space_samples)
       generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
       all_samples = torch.cat((real_samples, generated_samples))
       all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))


       # Training the discriminator
       discriminator.zero_grad()
       output_discriminator = discriminator(all_samples)
       loss_discriminator = loss_function(output_discriminator, all_samples_labels)
       loss_discriminator.backward()
       optimizer_discriminator.step()


       # Data for training the generator
       latent_space_samples = torch.randn((batch_size, 100)).to(device=device)


       # Training the generator
       generator.zero_grad()
       generated_samples = generator(latent_space_samples)
       output_discriminator_generated = discriminator(generated_samples)
       loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
       loss_generator.backward()
       optimizer_generator.step()


       # Print loss for every batch (optional)
       if n == len(train_loader) - 1:
           print(f"Epoch: {epoch+1}/{num_epochs}, Loss D.: {loss_discriminator.item()}, Loss G.: {loss_generator.item()}")


# Generate some new samples after training
latent_space_samples = torch.randn(batch_size, 100).to(device=device)
generated_samples = generator(latent_space_samples)


# Plot generated images
generated_samples = generated_samples.cpu().detach()
for i in range(16):
   ax = plt.subplot(4, 4, i + 1)
   plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
   plt.xticks([])
   plt.yticks([])


plt.show()
