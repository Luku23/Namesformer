import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import streamlit as st
import pandas as pd

# Adjusted NameDataset
class NameDataset(Dataset):
    def __init__(self, csv_file):
        self.names = pd.read_csv(csv_file)['name'].values # Load names from file
        self.chars = sorted(list(set(''.join(self.names) + ' ')))  # Including a padding character
        # Char to int and Int to char mappings
        self.char_to_int = {c: i for i, c in enumerate(self.chars)} # Char to int mapping
        self.int_to_char = {i: c for c, i in self.char_to_int.items()} # Int to char mapping
        self.vocab_size = len(self.chars) # Number of unique characters

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx] + ' '  # Adding padding character at the end
        encoded_name = [self.char_to_int[char] for char in name]
        return torch.tensor(encoded_name)

# Minimal Transformer Model
class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion):
        super(MinimalTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x

dataset_men = NameDataset("names_men.txt")
model_men = torch.load("model_men_final.pth")
model_men.eval()

# Load female dataset and model
dataset_women = NameDataset("names_women.txt")
model_women = torch.load("model_women_final.pth")
model_women.eval()


def sample(model, dataset, start_str='A', max_length=20, eos_token=' '):
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        # Convert start string to tensor
        chars = [dataset.char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)  # Add batch dimension

        output_name = start_str
        for _ in range(max_length - len(start_str)):
            output = model(input_seq)

            # Get the last character from the output
            probabilities = torch.softmax(output[0, -1], dim=0)
            # Sample a character from the probability distribution
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.int_to_char[next_char_idx]

            if next_char == eos_token:  # Assume ' ' is your end-of-sequence character
                break

            output_name += next_char
            # Update the input sequence for the next iteration
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)

        return output_name

# Streamlit App
st.title("Lithuanian Name Generator")
st.write("Enter a starting string, select the model, and generate a Lithuanian name!")

# User Input
start_str = st.text_input("Enter starting string:", value="A")
gender = st.radio("Choose gender:", ("Male", "Female"))

# Generate Name
if st.button("Generate Name"):
    if gender == "Male":
        generated_name = sample(model_men, dataset_men, start_str=start_str)
    else:
        generated_name = sample(model_women, dataset_women, start_str=start_str)

    st.write(f"Generated Name: **{generated_name}**")