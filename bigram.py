import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

block_size = 8
batch_size = 32
max_iters = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-2
eval_iters = 200
n_embd = 32


class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Read input data
with open("input.txt") as f:
    text = f.read()

# all unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create simple character tokenizer
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Encode input data to tensor and split to train and val splits
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


# Training loop
for step in range(max_iters):
    # Every eval_iters evaluate on training and validation splits
    if step % eval_iters == 0:
        losses = estimate_loss()
        print(
            f'step {step}: train loss: {losses["train"]:.4f}'
            f' val loss: {losses["val"]:.4f}'
        )

    # Sample data
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate from model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
