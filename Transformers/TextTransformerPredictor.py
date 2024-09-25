import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

class PositionalEncoding(nn.Module): 
    def __init__(self,d_model,max_len=5000):
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(max_len,d_model) 
        position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000)/d_model)) 


        pe[:,0::2] = torch.sin(position * div_term)

        pe[:,1::2] = torch.cos(position*div_term)

        pe=pe.unsqueeze(0) 
        self.register_buffer('pe',pe)


    def forward(self,x):
        return x+self.pe[:,:x.size(1)]
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads 
        self.num_heads = num_heads

        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model,d_model)
        self.fc=nn.Linear(d_model,d_model)

        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)
    def forward(self,query,key,value,mask=None):
        batch_size = query.size(0)

        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0 , -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn,value).transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.d_k)


        output = self.fc(context) 
        output = self.proj_dropout(output) 

        return output 


class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout = 0.1):
        super(FeedForward, self).__init__()
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))    


class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout = 0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.d_model = d_model
    def forward(self,x, src_mask=None):
        attn_output =  self.self_attn(x,x,x,src_mask)
        x = x + self.dropout1(attn_output)
        x= self.norm1(x)

        ffn_output = self.ffn(x)
        x = x+self.dropout2(ffn_output)
        x = self.norm2(x)

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self,encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layer for _ in range(num_layers))
        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self,src,mask=None):
        output = src 
        for layer in self.layers:
            output = layer(output,mask)
        return self.norm(output)
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, d_ff, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model, num_heads, d_ff, dropout), num_encoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        encoder_output = self.encoder(src, src_mask)
        output = self.fc_out(encoder_output)
        return output
    



sentences = [
    "The quick brown fox jumps over the lazy dog",
    "To be or not to be that is the question",
    "All that glitters is not gold",
    "A journey of a thousand miles begins with a single step",
    "Actions speak louder than words",
    "Beauty is in the eye of the beholder",
    "Every cloud has a silver lining",
    "Fortune favors the bold",
    "Honesty is the best policy",
    "Knowledge is power ignorance is bliss",
    "Laughter is the best medicine",
    "Practice makes perfect",
    "Rome wasn't built in a day",
    "The early bird catches the worm",
    "Time heals all wounds",
    "Where there's smoke there's fire",
    "You can't judge a book by its cover",
    "A penny saved is a penny earned",
    "Don't count your chickens before they hatch",
    "The pen is mightier than the sword",
    "When in Rome do as the Romans do",
    "An apple a day keeps the doctor away",
    "Better late than never",
    "Curiosity killed the cat",
    "Don't put all your eggs in one basket",
    "Every dog has its day",
    "Fools rush in where angels fear to tread",
    "Great minds think alike",
    "Haste makes waste",
    "If the shoe fits wear it",
    "Keep your friends close and your enemies closer",
    "Let sleeping dogs lie",
    "Money doesn't grow on trees",
    "No pain no gain",
    "Once bitten twice shy",
    "Pride comes before a fall",
    "The grass is always greener on the other side",
    "United we stand divided we fall",
    "Variety is the spice of life",
    "When it rains it pours",
    "You can lead a horse to water but you can't make it drink",
    "A picture is worth a thousand words",
    "Absence makes the heart grow fonder",
    "Beggars can't be choosers",
    "Don't bite the hand that feeds you",
    "Every rose has its thorn",
    "Fortune favors the prepared mind",
    "Good things come to those who wait",
    "Home is where the heart is",
    "It takes two to tango",
    "Jack of all trades master of none",
    "Laughter is the shortest distance between two people",
    "Necessity is the mother of invention",
    "One man's trash is another man's treasure",
    "Patience is a virtue",
    "The apple doesn't fall far from the tree",
    "Two wrongs don't make a right",
    "When the going gets tough the tough get going",
    "You reap what you sow",
    "It was the best of times it was the worst of times",
    "Call me Ishmael",
    "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife",
    "Happy families are all alike every unhappy family is unhappy in its own way",
    "In a hole in the ground there lived a hobbit",
    "It was a bright cold day in April and the clocks were striking thirteen",
    "All children except one grow up",
    "The man in black fled across the desert and the gunslinger followed",
    "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer",
    "Someone must have slandered Josef K for one morning without having done anything truly wrong he was arrested",
    "The past is a foreign country they do things differently there",
    "Whether I shall turn out to be the hero of my own life or whether that station will be held by anybody else these pages must show",
    "If you really want to hear about it the first thing you'll probably want to know is where I was born",
    "Many years later as he faced the firing squad Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice",
    "I am an invisible man",
    "The sky above the port was the color of television tuned to a dead channel",
    "Mrs Dalloway said she would buy the flowers herself",
    "All this happened more or less",
    "They shoot the white girl first",
    "For a long time I went to bed early",
    "The moment one learns English one starts to become a stranger to one's own self",
    "To the red country and part of the gray country of Oklahoma the last rains came gently",
    "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since",
    "You better not never tell nobody but God",
    "It was a pleasure to burn",
    "how are you? fine how are you?",
    ' DNA molecule consists of two long polynucleotide chains composed of four types of nucleotide subunits. Each of these chains is known as a DNA chain, or a DNA strand. Hydrogen bonds between the base portions of the nucleotides hold the two chains together'
]

# Tokenize 

words = set() ## empty set 

for sentence in sentences:
    words.update(sentence.lower().split()) #splits into words, lowercase and adds to set 

# vocabulary creation 
# creates a cocabulary (list of words) by
#   -adding two special tokens at the start of the covab list
#       - <pad> : a padding token used to pad sentences to teh same length 
#       - <unk> : unknown token, used for words that are not in vocab
vocab = ['<pad>','<unk>'] + sorted(list(words))


#creates a dict which maps each word in the vocab to a unique index (int)
#enumerate generates pairs of index and word, where idx is the index and word is the word from vocab
"""
word_to_idx = {
    '<pad>': 0,
    '<unk>': 1,
    'hello': 2,
    'machine': 3,
    'world': 4
}
idx_to_word = {
    0: '<pad>',
    1: '<unk>',
    2: 'hello',
    3: 'machine',
    4: 'world'
}
"""
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

vocab_size = len(vocab)


def prepare_sequence(seq):
    return [word_to_idx.get(word.lower(), word_to_idx['<unk>']) for word in seq.split()]

X = [prepare_sequence(sent) for sent in sentences]
y = [x[1:] + [word_to_idx['<pad>']] for x in X]  

max_len = max(len(seq) for seq in X)
X = [seq + [word_to_idx['<pad>']] * (max_len - len(seq)) for seq in X]
y = [seq + [word_to_idx['<pad>']] * (max_len - len(seq)) for seq in y]


X = torch.tensor(X)
y = torch.tensor(y)

d_model = 64  
num_heads = 4  
num_encoder_layers = 3  
d_ff = 256 
dropout = 0.1


model = Transformer(vocab_size, d_model, num_heads, num_encoder_layers, d_ff, dropout, max_len)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training 

num_epochs = 200
for epoch in range(num_epochs):
    model.train() 
    optimizer.zero_grad() 
    output = model(X) 
    loss=criterion(output.view(-1, vocab_size), y.view(-1)) 
    loss.backward() 
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

def predict_next_word(model, input_text, word_to_idx, idx_to_word):
    model.eval()
    with torch.no_grad():
        input_seq = prepare_sequence(input_text)
        input_tensor = torch.tensor([input_seq]).long()
        output = model(input_tensor)
        last_word_logits = output[0, -1, :]
        predicted_idx = torch.argmax(last_word_logits).item()
        return idx_to_word[predicted_idx]
    
test_sentences = [
    "The quick brown fox",
    "To be or not to",
    "All that glitters is",
    "Actions speak louder than",
    "Knowledge is power ignorance",
    "It was the best of",
    "Call me",
    "The past is a foreign",
    "Whether I shall turn out",
    "The sky above the"
]

print("\nPredictions:")
for sentence in test_sentences:
    next_word = predict_next_word(model, sentence, word_to_idx, idx_to_word)
    print(f"Input: '{sentence}' | Predicted next word: '{next_word}'")


def get_word_lengths(sentence):
    words = sentence.split()
    return len(words)

print("\nInteractive Prediction:")
while True:
    user_input = input("Enter a phrase (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    next_word = user_input+" "+ predict_next_word(model, user_input, word_to_idx, idx_to_word)
    while (get_word_lengths(next_word)< 20):
        next_word = next_word+ " "+ predict_next_word(model, next_word, word_to_idx, idx_to_word)
    print(f"Predicted next word: '{next_word}'")   