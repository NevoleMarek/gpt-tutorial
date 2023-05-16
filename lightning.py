import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"


class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size) -> None:
        super().__init__()
        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class Head(nn.Module):
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size, n_embd, dropout, block_size)
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head, head_size, n_embd, dropout, block_size
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(pl.LightningModule):
    def __init__(
        self, n_embd, dropout, n_head, n_layer, vocab_size, block_size
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, dropout, block_size)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y = y.view(B * T)

        loss = F.cross_entropy(logits, y)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y = y.view(B * T)

        loss = F.cross_entropy(logits, y)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


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


def main():
    block_size = 128
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = 3e-4
    n_embd = 128
    dropout = 0.2
    n_head = 32
    n_layer = 32

    # Data
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

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = ShakespeareDataset(data[:n], block_size)
    val_data = ShakespeareDataset(data[n:], block_size)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=6)

    model = BigramLanguageModel(
        n_embd=n_embd,
        dropout=dropout,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        vocab_size=vocab_size,
    )
    m = model.to(device)
    trainer = pl.Trainer(
        limit_train_batches=2000,
        limit_val_batches=10,
        val_check_interval=100,
        max_epochs=1,
        devices=1,
        accelerator="gpu",
        precision=16,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # Generate from model
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
