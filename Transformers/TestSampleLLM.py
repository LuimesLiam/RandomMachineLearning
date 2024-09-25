import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_text(text):
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)

with open('Transformers/train.txt', 'r') as file:
    text = file.read()

encoded_text = encode_text(text)

class TextDataset(Dataset):
    def __init__(self, encoded_text):
        self.data = torch.tensor(encoded_text, dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        return self.data[idx:idx+2]

dataset = TextDataset(encoded_text)

def collate_fn(batch):
    input_ids = torch.stack([item[:-1] for item in batch])
    labels = torch.stack([item[1:] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

def train(model, dataloader, optimizer, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

train(model, dataloader, optimizer, device)

def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits[:, -1, :]
            predicted_token = torch.argmax(predictions, dim=-1)
            
            if predicted_token.item() == tokenizer.sep_token_id:
                break
            
            input_ids = torch.cat([input_ids, predicted_token.unsqueeze(0)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Example usage
response = generate_response('Machine learning is')
print(response)