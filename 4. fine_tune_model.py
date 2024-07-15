# fine_tune_model.py
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length)
        return torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask'])

def fine_tune(model, tokenizer, dataset, epochs=1, batch_size=2, lr=5e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask in dataloader:
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch+1} completed. Loss: {loss.item()}')

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    texts = ["Example text for fine-tuning the GPT-2 model."] * 100  # Example texts for fine-tuning
    
    dataset = TextDataset(tokenizer, texts, max_length=50)
    fine_tune(model, tokenizer, dataset, epochs=3, batch_size=2)
    
    model.save_pretrained('./fine_tuned_gpt2')
    tokenizer.save_pretrained('./fine_tuned_gpt2')
    print("Model fine-tuning completed and saved.")
