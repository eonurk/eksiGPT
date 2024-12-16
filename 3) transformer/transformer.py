import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

# hyperparameters
kume = 64 # how many independent sequences will we process in parallel?
pencere = 256 # what is the maximum context length for predictions?
maksimum_iterasyon = 5000
degerlendirme_iterasyon = 500
ogrenme_orani = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
gomme_boyutu = 384
dikkat_buyuklugu = 6
katman_sayisi = 6
dropout = 0.2
use_mixed_precision = device.type == 'cuda'  # Only use mixed precision on GPU
num_latents = 64  # number of latent vectors
d_latent = gomme_boyutu  # dimension of latent vectors
# ------------

torch.manual_seed(1337)

with open('../data/eksi_articles.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
karakter = sorted(list(set(text)))
sozluk_boyutu = len(karakter)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(karakter) }
itos = { i:ch for i,ch in enumerate(karakter) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - pencere, (kume,))
    x = torch.stack([data[i:i+pencere] for i in ix])
    y = torch.stack([data[i+1:i+pencere+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, gomme_boyutu):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gomme_boyutu, 4 * gomme_boyutu),
            nn.ReLU(),
            nn.Linear(4 * gomme_boyutu, gomme_boyutu),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PerceiverCrossAttention(nn.Module):
    def __init__(self, d_model, d_latent, num_latents, num_heads=1):
        super().__init__()
        self.num_latents = num_latents
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_heads = num_heads

        # Latents are learnable parameters, independent of input length
        self.latents = nn.Parameter(torch.randn(num_latents, d_latent))

        # Projection layers for Q (from latents), K and V (from input)
        self.q_proj = nn.Linear(d_latent, d_latent)
        self.k_proj = nn.Linear(d_model, d_latent)
        self.v_proj = nn.Linear(d_model, d_latent)

        # Optional: A linear layer to combine heads if using multi-head attention
        if num_heads > 1:
            # Ensure d_latent is divisible by num_heads
            assert d_latent % num_heads == 0, "d_latent must be divisible by num_heads"
            self.head_dim = d_latent // num_heads
            self.out_proj = nn.Linear(d_latent, d_latent)

    def forward(self, x):
        """
        x: Input tensor of shape (batch, N, d_model)
        returns: updated latents of shape (batch, num_latents, d_latent)
        """
        B, N, _ = x.size()

        # Expand latents for batch dimension
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_latent)

        # Compute Q, K, V
        Q = self.q_proj(latents)  # (B, M, d_latent)
        K = self.k_proj(x)        # (B, N, d_latent)
        V = self.v_proj(x)        # (B, N, d_latent)

        # If multi-head attention, reshape Q, K, V to (B, heads, seq, head_dim)
        if self.num_heads > 1:
            Q = Q.reshape(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, M, head_dim)
            K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)               # (B, heads, N, head_dim)
            V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)               # (B, heads, N, head_dim)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, heads, M, N)

            # Softmax over the last dimension (N)
            attn = F.softmax(scores, dim=-1)

            # Compute attention output
            out = torch.matmul(attn, V)  # (B, heads, M, head_dim)

            # Combine heads
            out = out.transpose(1, 2).reshape(B, self.num_latents, self.d_latent)  # (B, M, d_latent)
            out = self.out_proj(out)

        else:
            # Single head attention
            scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_latent ** 0.5) # (B, M, N)
            attn = F.softmax(scores, dim=-1)   # (B, M, N)
            out = torch.matmul(attn, V)        # (B, M, d_latent)

        # 'out' now is a representation of fixed-size M, integrating info from all N tokens
        return out



class Block(nn.Module):
    """ Transformer block with Perceiver Cross-Attention """

    def __init__(self, gomme_boyutu, dikkat_buyuklugu):
        super().__init__()
        self.perceiver = PerceiverCrossAttention(
            d_model=gomme_boyutu,
            d_latent=d_latent,
            num_latents=num_latents,
            num_heads=dikkat_buyuklugu
        )
        self.ffwd = FeedFoward(gomme_boyutu)
        self.ln1 = nn.LayerNorm(gomme_boyutu)
        self.ln2 = nn.LayerNorm(gomme_boyutu)
        
        # Additional projection to match dimensions
        self.output_proj = nn.Linear(num_latents * d_latent, pencere * gomme_boyutu)

    def forward(self, x):
        B, T, C = x.shape
        
        # Apply layer norm before perceiver
        x_norm = self.ln1(x)
        
        # Pass through perceiver
        latent_output = self.perceiver(x_norm)  # (B, num_latents, d_latent)
        
        # Reshape and project back to original sequence length
        latent_flat = latent_output.reshape(B, -1)  # (B, num_latents * d_latent)
        output = self.output_proj(latent_flat)  # (B, pencere * gomme_boyutu)
        output = output.reshape(B, T, C)  # (B, T, C)
        
        # Add residual connection
        x = x + output
        
        # Feed forward part remains the same
        x = x + self.ffwd(self.ln2(x))
        return x

class EksiGPT(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(sozluk_boyutu, gomme_boyutu)
        self.position_embedding_table = nn.Embedding(pencere, gomme_boyutu)
        self.blocks = nn.Sequential(*[Block(gomme_boyutu, dikkat_buyuklugu=dikkat_buyuklugu) for _ in range(katman_sayisi)])
        self.ln_f = nn.LayerNorm(gomme_boyutu) # final layer norm
        self.lm_head = nn.Linear(gomme_boyutu, sozluk_boyutu)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,sozluk_boyutu)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last pencere tokens
            idx_cond = idx[:, -pencere:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = EksiGPT()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=ogrenme_orani)

# Create gradient scaler for mixed precision
scaler = GradScaler(enabled=use_mixed_precision)

for iter in range(maksimum_iterasyon):
    if iter % degerlendirme_iterasyon == 0 or iter == maksimum_iterasyon - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))