import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(52 * 768, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 256),  # Starting from a 100-dim latent space
            nn.ReLU(),
            nn.Linear(256, 52 * 768),  # Adjust the output size to match the flattened shape
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, 52, 768)  # Reshape to the required output shape


def train_gan_with_hinge_loss(generator, discriminator, train_dataset, test_dataset, model_path):
    # Hyperparameters
    learning_rate = 0.0002
    batch_size = 32
    epochs = 3
    generator_name = 'hinge_gan_generator'
    discriminator_name = 'hinge_gan_discriminator'

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (real_data, _) in enumerate(train_loader):
            current_batch_size = real_data.size(0)
            real_data = real_data.view(current_batch_size, -1)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_loss = torch.mean(F.relu(1.0 - discriminator(real_data)))
            z = torch.randn(current_batch_size, 768)
            fake_data = generator(z)
            fake_loss = torch.mean(F.relu(1.0 + discriminator(fake_data)))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            z = torch.randn(current_batch_size, 768)
            fake_data = generator(z)
            g_loss = -torch.mean(discriminator(fake_data))
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    # Save models
    torch.save(generator, os.path.join(model_path, f'{generator_name}.pth'))
    torch.save(discriminator, os.path.join(model_path, f'{discriminator_name}.pth'))

def train_lsgan(generator, discriminator, train_dataset, test_dataset, model_path):
    # Hyperparameters
    learning_rate = 0.0002
    batch_size = 32
    epochs = 3
    generator_name = 'least_square_generator'
    discriminator_name = 'least_square_discriminator'

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (real_data, _) in enumerate(train_loader):
            current_batch_size = real_data.size(0)
            real_data = real_data.view(current_batch_size, -1)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(real_data), torch.ones_like(discriminator(real_data)))
            z = torch.randn(current_batch_size, 768)
            fake_data = generator(z)
            fake_loss = criterion(discriminator(fake_data), torch.zeros_like(discriminator(fake_data)))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            z = torch.randn(current_batch_size, 768)
            fake_data = generator(z)
            g_loss = criterion(discriminator(fake_data), torch.ones_like(discriminator(fake_data)))
            g_loss.backward()
            optimizer_g.step()

            print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    # Save models
    torch.save(generator, os.path.join(model_path, f'{generator_name}.pth'))
    torch.save(discriminator, os.path.join(model_path, f'{discriminator_name}.pth'))

def train_wasserstein_gan(generator, discriminator, train_dataset, test_dataset, model_path):
    # Hyperparameters
    learning_rate = 0.0002
    batch_size = 32  # Ensure this is consistent with your DataLoader
    epochs = 3
    n_critic = 5
    weight_clip = 0.01
    generator_name = 'wasserstein_loss_generator'
    discriminator_name = 'wasserstein_loss_discriminator'

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Assuming validation_loader is needed
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (real_data, _) in enumerate(train_loader):
            current_batch_size = real_data.size(0)

            # Flatten the real data (if necessary)
            real_data = real_data.view(current_batch_size, -1)

            # Update the Critic (Discriminator)
            optimizer_d.zero_grad()
            real_loss = discriminator(real_data).mean()
            z = torch.randn(current_batch_size, 768)
            fake_data = generator(z)
            fake_loss = discriminator(fake_data).mean()
            d_loss = fake_loss - real_loss
            d_loss.backward()
            optimizer_d.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

            # Update the Generator
            if i % n_critic == 0:
                optimizer_g.zero_grad()
                # Regenerate fake data for generator update
                z = torch.randn(current_batch_size, 768)
                fake_data = generator(z)
                g_loss = -discriminator(fake_data).mean()
                g_loss.backward()
                optimizer_g.step()

        # Logging, etc.
        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    torch.save(generator, os.path.join(model_path, f'{generator_name}.pth'))
    torch.save(discriminator, os.path.join(model_path, f'{discriminator_name}.pth'))
