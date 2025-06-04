import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import torch.optim as optim
from torch.utils.data import random_split
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



torch.manual_seed(1337)
# hyperparameters
num_heads = 8 
num_layer = 6
dropout = 0.2
n_embed = 512
max_seq_len = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 10  # Increased epochs
patience = 3      # For early stopping
epochs_no_improve = 0
best_val_loss = float('inf')
print_every = 100
accumulation_steps = 4  # Gradient accumulation steps
lr=3e-4 * accumulation_steps  # Adjusted learning rate for accumulation
smoothie = SmoothingFunction().method4


# loading dataset for training
# Load the raw sentence pairs
def load_sentence_pairs(path, num_examples=None):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
    pairs = [[l.split("\t")[0], l.split("\t")[1]] for l in lines]
    if num_examples:
        pairs = pairs[:num_examples]
    return pairs

pairs = load_sentence_pairs("transformer/deu.txt", num_examples=50000)  # Toy size

def tokenize(text):
    text = text.lower()
    # Keep punctuation for better context
    return re.findall(r"\w+|[^\w\s]", text)

def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenize(sentence))
    
    vocab = ["<pad>", "<sos>", "<eos>", "<unk>"]
    vocab += [word for word, freq in counter.items() if freq >= min_freq]
    
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    
    return vocab, stoi, itos

src_sentences = [s[0] for s in pairs]
tgt_sentences = [s[1] for s in pairs]

src_vocab, src_stoi, src_itos = build_vocab(src_sentences)
tgt_vocab, tgt_stoi, tgt_itos = build_vocab(tgt_sentences)

vocab_size = max(len(src_vocab), len(tgt_vocab))  # Set for your model


def numericalize(sentence, stoi, max_len=max_seq_len):
    tokens = ["<sos>"] + tokenize(sentence)[:max_len-2] + ["<eos>"]
    ids = [stoi.get(tok, stoi["<unk>"]) for tok in tokens]
    return ids + [stoi["<pad>"]] * (max_len - len(ids))


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_stoi, tgt_stoi, max_len=max_seq_len):
        self.pairs = pairs
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = numericalize(src, self.src_stoi, self.max_len)
        tgt_ids = numericalize(tgt, self.tgt_stoi, self.max_len)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

dataset = TranslationDataset(pairs, src_stoi, tgt_stoi)

# Split into train and validation sets
train_size = int(0.9 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False)





# Transformer Implementation from research paper - Attention is all you needs
class ScaledDotProduct(nn.Module):
    """ Self-Attention """
    
    def __init__(self, H_size):
        super().__init__()
        self.scale = H_size ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # B : Batch size, heads : Number of heads, t_q : sequence lenght, H : Size of each head
        # q: (B, heads, T_q, H)
        # k: (B, heads, T_k, H)
        # v: (B, heads, T_k, H)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T_q, T_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v  # (B, heads, T_q, H)
        return out
    
    
class MultiHeadAttention(nn.Module):
    """ Multi-Head-Attention """
    
    def __init__(self, H_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.H_size = H_size

        self.q_linear = nn.Linear(n_embed, H_size * num_heads)
        self.k_linear = nn.Linear(n_embed, H_size * num_heads)
        self.v_linear = nn.Linear(n_embed, H_size * num_heads)
        self.out_proj = nn.Linear(H_size * num_heads, n_embed)

        self.attn = ScaledDotProduct(H_size)
        
        self.dropout = nn.Dropout(dropout)

    # def forward(self, q, k, v):  # q, k, v shape: (B, T, C)
    #     B, T_q, _ = q.shape
    #     B, T_k, _ = k.shape

    #     # Project and split heads
    #     q = self.q_linear(q).view(B, T_q, self.num_heads, self.H_size).transpose(1, 2)  # (B, heads, T_q, H)
    #     k = self.k_linear(k).view(B, T_k, self.num_heads, self.H_size).transpose(1, 2)  # (B, heads, T_k, H)
    #     v = self.v_linear(v).view(B, T_k, self.num_heads, self.H_size).transpose(1, 2)  # (B, heads, T_k, H)

    #     out = self.attn(q, k, v)  # (B, heads, T_q, H)
    #     out = out.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.H_size)  # (B, T_q, C) ---> (B, T_q, C)
    #     out = self.dropout(self.out_proj(out))
    #     return out  # (B, T_q, C)
    
    def forward(self, q, k=None, v=None):
        # Handle both positional and keyword arguments
        if k is None and v is None:
            # Assume self-attention if only q provided
            k = v = q
        elif v is None:
            # If v not provided but k is, use k for v
            v = k
        
        B, T_q, _ = q.shape
        B, T_k, _ = k.shape

        # Project and split heads
        q = self.q_linear(q).view(B, T_q, self.num_heads, self.H_size).transpose(1, 2)
        k = self.k_linear(k).view(B, T_k, self.num_heads, self.H_size).transpose(1, 2)
        v = self.v_linear(v).view(B, T_k, self.num_heads, self.H_size).transpose(1, 2)

        out = self.attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.H_size)
        out = self.dropout(self.out_proj(out))
        return out
    
class FeedForwardLayer(nn.Module):
    """ Simple linear layer """
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
    
class EncoderBlock(nn.Module):
    """Single Encoder Block"""
    
    def __init__(self, num_heads, n_embed):
        super().__init__()
        H_size = n_embed // num_heads
        self.sa = MultiHeadAttention(H_size, num_heads)
        self.ffnn = FeedForwardLayer(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffnn(x))
        return x
    

class TransformerEncoder(nn.Module):
    """Full Encoder Block"""
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(max_seq_len, n_embed)
        self.blocks = nn.Sequential(*[EncoderBlock(num_heads, n_embed) for _ in range(num_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T,C)
        pos_emb = pos_emb.unsqueeze(0)  # (1, T, C), broadcastable to (B, T, C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.dropout(x)
        x = self.blocks(x) # (B,T,C)
        x = self.ln(x) # (B,T,C)        
        return x
    
    
class MaskedScaledDotProduct(nn.Module):
    """ Self-Attention with mask"""
    
    def __init__(self, H_size):
        super().__init__()
        self.query = nn.Linear(n_embed, H_size, bias= False)
        self.key = nn.Linear(n_embed, H_size, bias= False)
        self.value = nn.Linear(n_embed, H_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) #(B, T, h_size)
        k = self.key(x) #(B, T, h_size)
        v = self.value(x) #(B, T, h_size)
        score = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, h_size) * (B, h_size, T) --> (B, T, T)/sqrt(h_size)
        mask = self.tril[:T, :T].unsqueeze(0)  # (1, T, T)
        score = score.masked_fill(mask == 0, float('-inf'))
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        out = score @ v # (B, T, T) ---> (B, T, h_size)
        return out


class MaskedMultiHeadAttention(nn.Module):
    """ Multi-Head-Attention with mask """
    
    def __init__(self, H_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([MaskedScaledDotProduct(H_size) for _ in range(num_heads)])
        self.proj = nn.Linear(H_size*num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
    
class DecoderBlock(nn.Module):
    """ Single Decoder Block """
    
    def __init__(self, num_heads, n_embed):
        super().__init__()
        H_size = n_embed // num_heads

        # 1. Masked Self-Attention
        self.masked_attn = MaskedMultiHeadAttention(H_size, num_heads)
        self.ln1 = nn.LayerNorm(n_embed)

        # 2. Encoder-Decoder Attention
        self.cross_attn = MultiHeadAttention(H_size, num_heads)
        self.ln2 = nn.LayerNorm(n_embed)

        # 3. Feed Forward
        self.ffnn = FeedForwardLayer(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)
        

    def forward(self, x, enc_out):
        # x: (B, T_dec, n_embed)
        # enc_out: (B, T_enc, n_embed)

        # 1. Masked Self-Attention
        x = self.ln1(x + self.masked_attn(x))  # (B, T_dec, n_embed)

        # 2. Encoder-Decoder (Cross) Attention
        # x = self.ln2(x + self.cross_attn(q=x, k=enc_out, v=enc_out))  # pass q, k, v explicitly
        x = self.ln2(x + self.cross_attn(x, enc_out, enc_out))  

        # 3. Feed Forward
        x = self.ln3(x + self.ffnn(x))  # (B, T_dec, n_embed)

        return x
    
    
class DecoderBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x_enc_tuple):
        x, enc_out = x_enc_tuple
        x = self.block(x, enc_out)
        return (x, enc_out)  # return tuple for next block
    
    
    
class TransformerDecoder(nn.Module):
    """ Full Decoder Block """
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(max_seq_len, n_embed)
        self.blocks = nn.Sequential(*[DecoderBlockWrapper(DecoderBlock(num_heads, n_embed)) for _ in range(num_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, enc_out):
        # idx: (B, T_dec)
        # enc_out: (B, T_enc, n_embed)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T_dec, n_embed)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=idx.device))  # (T_dec, n_embed)
        pos_emb = pos_emb.unsqueeze(0)  # (1, T_dec, n_embed)

        x = tok_emb + pos_emb  # (B, T_dec, n_embed)
        x = self.dropout(x)
        x, _ = self.blocks((x, enc_out))   # pass encoder output to each DecoderBlock
        x = self.ln(x)
        return x  # (B, T_dec, n_embed)
    
    
    
class Transformer(nn.Module):
    """ Final Transformer """
    
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, src_idx, tgt_idx):
        # src_idx: (B, T_enc)
        # tgt_idx: (B, T_dec)
        enc_out = self.encoder(src_idx)               # (B, T_enc, n_embed)
        dec_out = self.decoder(tgt_idx, enc_out)      # (B, T_dec, n_embed)
        logits = self.lm_head(dec_out)                # (B, T_dec, vocab_size)
        return logits
    
    
# Model loading
model = Transformer().to(device)

# Loss with label smoothing
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tgt_stoi["<pad>"],
    label_smoothing=0.1  # Added label smoothing
)

# Optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_tokens = 0
    optimizer.zero_grad()
    
    for step, (src, tgt) in enumerate(train_dataloader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        
        logits = model(src, tgt_input)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        num_tokens = (tgt_target != tgt_stoi["<pad>"]).sum()
        total_loss += loss.item() * num_tokens * accumulation_steps
        total_tokens += num_tokens
        
        if (step + 1) % print_every == 0:
            avg_loss = total_loss / total_tokens
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], "
                f"Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
            total_loss = 0
            total_tokens = 0

    if len(train_dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    # === Validation ===
    model.eval()
    val_loss = 0
    val_tokens = 0
    bleu_predictions = []
    bleu_references = []

    with torch.no_grad():
        for val_src, val_tgt in valid_dataloader:
            val_src, val_tgt = val_src.to(device), val_tgt.to(device)
            val_tgt_input = val_tgt[:, :-1]
            val_tgt_target = val_tgt[:, 1:]
            
            val_logits = model(val_src, val_tgt_input)
            loss_val = loss_fn(
                val_logits.reshape(-1, val_logits.size(-1)),
                val_tgt_target.reshape(-1)
            )
            num_val_tokens = (val_tgt_target != tgt_stoi["<pad>"]).sum()
            val_loss += loss_val.item() * num_val_tokens
            val_tokens += num_val_tokens

            # --- BLEU Computation ---
            for i in range(val_src.size(0)):
                src_sample = val_src[i].unsqueeze(0)
                pred_tokens = [tgt_stoi["<sos>"]]
                decoder_input = torch.tensor([[tgt_stoi["<sos>"]]], device=device)
                for _ in range(50):
                    logits = model(src_sample, decoder_input)
                    next_token = logits.argmax(-1)[:, -1]
                    pred_tokens.append(next_token.item())
                    if next_token.item() == tgt_stoi["<eos>"]:
                        break
                    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

                pred_sentence = [tgt_itos[idx] for idx in pred_tokens if idx not in {tgt_stoi["<pad>"], tgt_stoi["<sos>"], tgt_stoi["<eos>"]}]
                ref_sentence = [tgt_itos[idx.item()] for idx in val_tgt[i] if idx.item() not in {tgt_stoi["<pad>"], tgt_stoi["<sos>"], tgt_stoi["<eos>"]}]
                
                if len(pred_sentence) > 0 and len(ref_sentence) > 0:
                    bleu_score = sentence_bleu([ref_sentence], pred_sentence, smoothing_function=smoothie)
                    bleu_predictions.append(bleu_score)

    avg_val_loss = val_loss / val_tokens
    avg_bleu = sum(bleu_predictions) / len(bleu_predictions) if bleu_predictions else 0.0
    print(f"\nEpoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, BLEU Score: {avg_bleu:.4f}")

    scheduler.step(avg_val_loss)
    print(f"Learning rate updated to: {optimizer.param_groups[0]['lr']:.2e}") 

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # === Sampling ===
    with torch.no_grad():
        random_idx = torch.randint(0, len(valid_dataset), (3,))
        for i in random_idx:
            src_sample, tgt_sample = valid_dataset[i]
            src_sample = src_sample.unsqueeze(0).to(device)
            pred_tokens = [tgt_stoi["<sos>"]]
            decoder_input = torch.tensor([[tgt_stoi["<sos>"]]], device=device)

            for _ in range(50):
                logits = model(src_sample, decoder_input)
                next_token = logits.argmax(-1)[:, -1]
                pred_tokens.append(next_token.item())
                if next_token.item() == tgt_stoi["<eos>"]:
                    break
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

            src_text = " ".join([src_itos.get(idx.item(), "<unk>") for idx in src_sample[0] if idx.item() != src_stoi["<pad>"]])
            tgt_text = " ".join([tgt_itos.get(idx.item(), "<unk>") for idx in tgt_sample if idx.item() != tgt_stoi["<pad>"]])
            pred_text = " ".join([tgt_itos.get(idx, "<unk>") for idx in pred_tokens if idx != tgt_stoi["<pad>"]])

            print(f"\nSample {i}:")
            print(f"SRC: {src_text}")
            print(f"TGT: {tgt_text}")
            print(f"PRED: {pred_text}")

    model.train()
