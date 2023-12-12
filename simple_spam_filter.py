import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(nb_words, emb_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(glove_embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # RNN Layer
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        x_embed = self.embedding(x)

        # Forward propagate the RNN
        out, _ = self.rnn(x_embed, h0)

        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        # print(out.shape)
        return out


from torch.utils.data import DataLoader

# Assuming `train_dataset` is your preprocessed and encoded training dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Hyperparameters
input_dim = 50  # This should match your encoding size
hidden_dim = 128
output_dim = 1   # Binary classification (Spam or Not Spam)
num_layers = 2
learning_rate = 0.001
num_epochs = 15
model_name = 'basic_rnn_spam_filter'
nb_words = NB_WORDS


def train_spam_filter(input_dim, hidden_dim, output_dim, num_layers, learning_rate, num_epochs,train_loader,validation_loader,model_name):
    # Initialize model, loss, and optimizer
    model = RNNModel(input_dim, hidden_dim, output_dim, num_layers)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'valid_loss': []
    }

    # Train the model
    for epoch in range(num_epochs):
      model.train()
      epoch_loss = 0
      for texts, labels in train_loader:
          # Forward pass
          outputs = model(texts)
          loss = loss_fn(outputs.squeeze(), labels.float())

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          #accumate loss
          epoch_loss += loss.item()

      # Average loss for the epoch
      epoch_loss /= len(train_loader)
      history['train_loss'].append(epoch_loss)

      # Validation phase
      model.eval()
      val_loss = 0
      with torch.no_grad():
          for texts, labels in validation_loader:
              # Forward pass
              pred = model(texts)
              loss = loss_fn(pred.squeeze(), labels.float())

              # Accumulate validation loss
              val_loss += loss.item()

      # Average validation loss for the epoch
      val_loss /= len(validation_loader)
      history['valid_loss'].append(val_loss)
      print('train:',epoch_loss,'valid:',val_loss)

    torch.save(model, os.path.join(model_path,f'{model_name}.pth'))

    # Epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # Training and validation loss
    train_loss = history['train_loss']
    valid_loss = history['valid_loss']

    # Create a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, valid_loss, label='Validation Loss', color='red')

    # Adding title and labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()
