import random
import pandas as pd
import numpy as np

def sent_parse(sentence, vocab, max_len):
    # Tokenize the sentence
    tokens = sentence.split()

    # Define special tokens
    pad_token_index = vocab.get("<pad>", 0)  # Default to 0 if "<pad>" is not in vocab
    unk_token_index = vocab.get("[UNK]", 1)  # Default to 1 if "<unk>" is not in vocab

    # Map tokens to their indices in the vocabulary
    token_indices = [vocab.get(token, unk_token_index) for token in tokens]

    # Pad or truncate the sequence to max_len
    padded_indices = token_indices[:max_len] + [pad_token_index] * max(0, max_len - len(token_indices))

    return torch.tensor(padded_indices, dtype=torch.long)

def encode_sentence(model, sentence, vocab, max_len):
    # Convert sentence to tensor of word indices
    sentence_tensor = sent_parse(sentence, vocab, max_len)  # Assuming sent_parse function exists
    sentence_tensor = sentence_tensor.unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        z_mean, _ = model.encode(sentence_tensor)
    return z_mean

input_text = """
𝐍𝐚𝐮𝐠𝐡𝐭𝐲-𝐆𝐢𝐫𝐥


I would love to find someone who will always be ready to fuck me when I need you to. Someone who enjoys anal and oral sex just as much as I do. I want a lover who can make me cum ov"""
encoded_sentence = encode_sentence(generator, input_text, index2word, max_len)

def create_random_latent_vector(batch_size=32, latent_dim=32, min_seq_len=10, max_seq_len=15):
    # Randomly choose a sequence length between min_seq_len and max_seq_len
    sequence_length = random.randint(min_seq_len, max_seq_len)

    # Create a random tensor with shape [batch_size, sequence_length, latent_dim]
    random_latent_vector = torch.randn(batch_size, sequence_length, latent_dim)

    return random_latent_vector

def process_text(text,emojis):
    # Replace multiple "num" with a single random number between 1-10000 (only keep the first two "num")
    num_count = 0
    def replace_num(match):
        nonlocal num_count
        num_count += 1
        return str(random.randint(1, 10000)) if num_count <= 2 else ''

    text = re.sub(r'\bnum\b', replace_num, text, flags=re.IGNORECASE)

    # Replace multiple "url" with "https://bitly.ly/" followed by random number (only keep the last one)
    urls = re.findall(r'\burl\b', text, flags=re.IGNORECASE)
    if urls:
        text = re.sub(r'\burl\b', '', text, flags=re.IGNORECASE, count=len(urls)-1)
        text = re.sub(r'\burl\b', f'https://bitly.ly/{random.randint(1, 10000)}', text, flags=re.IGNORECASE)

    # Remove repeated words (case insensitive)
    seen = set()
    words = text.split()
    result_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower not in seen:
            result_words.append(word)
            seen.add(word_lower)
    text = ' '.join(result_words)

    text = re.sub(r'\bunk\b', random.choice(emojis), text, flags=re.IGNORECASE)

    return text

def generate_sentence(model, size, vocab,emojis,batch_size=32,latent_dim=32,min_seq_len=5,max_seq_len=50):
    # latent_vector = torch.tensor(latent_vector).float().unsqueeze(0)  # Add batch dimension
    # print(latent_vector.shape)
    count = 0
    sentences = []
    while count < size:
      latent_vector = create_random_latent_vector(batch_size=batch_size,latent_dim=latent_dim,min_seq_len=min_seq_len,max_seq_len=max_seq_len)

      model.eval()
      with torch.no_grad():
          reconstructed_logits = model.decode(latent_vector)
      # print(reconstructed_logits.shape)

      # Convert logits to word indices
      word_indices = torch.argmax(F.softmax(reconstructed_logits, dim=-1), dim=-1)

      # Convert indices to words
      for seq in word_indices:
          sentence = [vocab.get(idx.item(), "") for idx in seq if idx.item() != 0]
          sentence = ' '.join(sentence)
          sentence = process_text(sentence,emojis)

          if len(sentence.split(' '))>min_seq_len:
              sentences.append(sentence)
              # print(sentence)
      count = len(sentences)
      # print(count)

    return sentences

def batch_generation(seeds, model, size=2,batch_size=32,latent_dim=32,min_seq_len=15,max_seq_len=50,emojis=emojis,vocab=index2word):
    sentences = []
    for seed in seeds['payload_text'].to_numpy().tolist():
        encoded_sentence = encode_sentence(model, seed, index2word, max_seq_len)
        generated_sentence = generate_sentence(model, size , index2word,emojis, 1)
        for item in generated_sentence:
            sentences.append(item)
    new_data = pd.DataFrame({'sentences': sentences})
    new_data['is_abusive'] = True
    return new_data
