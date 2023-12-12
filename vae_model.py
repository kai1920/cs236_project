import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, nb_words, max_len, emb_dim, intermediate_dim, latent_dim, glove_embedding_matrix):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(nb_words, emb_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(glove_embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(emb_dim, intermediate_dim // 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.dense = nn.Linear(intermediate_dim, intermediate_dim)
        self.dropout = nn.Dropout(0.2)
        self.z_mean = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(intermediate_dim, latent_dim)

        # Decoder layers
        self.decoder_h = nn.LSTM(latent_dim, intermediate_dim // 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.decoder_mean = nn.Linear(intermediate_dim, nb_words)  # TimeDistributed in PyTorch is implicit

    def encode(self, x):
        x_embed = self.embedding(x)
        h, _ = self.lstm(x_embed)
        h = self.dropout(F.elu(self.dense(h)))
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        h_decoded, _ = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)
        return x_decoded_mean

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, z_mean, z_log_var

#sequence loss function
def sequence_loss(logits, targets, reduction='mean'):
    """
    Computes the sequence loss

    Args:
    logits: Tensor of shape [batch_size, sequence_length, num_classes]
    targets: Tensor of shape [batch_size, sequence_length] with class indices
    reduction: 'mean' or 'sum', how to aggregate the loss

    Returns:
    Scalar loss value
    """
    batch_size, sequence_length, num_classes = logits.size()

    # Reshape for calculating cross-entropy loss
    logits_flat = logits.view(-1, num_classes)  # [batch_size * sequence_length, num_classes]
    targets_flat = targets.view(-1).long()  # [batch_size * sequence_length]
    # Compute cross entropy loss for each time step
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    # Reshape loss back to [batch_size, sequence_length]
    loss = loss.view(batch_size, sequence_length)

    # Aggregate the loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError("Invalid reduction type. Choose 'mean' or 'sum'.")

# Loss Function
def vae_loss_function(x_decoded_mean, x, z_mean, z_log_var):
    # Reconstruction loss
    xent_loss = sequence_loss(x_decoded_mean, x, reduction='mean')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
    # print('worked')
    return xent_loss + kl_loss
