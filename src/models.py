"""1D ResNet ECG encoder, linear head (baseline) and text-anchored label-query head (proposed)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixStyle1d(nn.Module):
    """Mixes per-sample channel statistics across the batch (Zhou et al., ICLR 2021), 1D version."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(2, keepdim=True)
        sig = (x.var(2, keepdim=True) + self.eps).sqrt()
        xn = (x - mu.detach()) / sig.detach()
        lam = self.beta.sample((B, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu2 = lam * mu + (1 - lam) * mu[perm]
        sig2 = lam * sig + (1 - lam) * sig[perm]
        return xn * sig2 + mu2


class Block(nn.Module):
    def __init__(self, cin, cout, stride, k=7):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(cout, cout // 8, 1), nn.ReLU(),
                                nn.Conv1d(cout // 8, cout, 1), nn.Sigmoid())
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        h = h * self.se(h)
        return F.relu(h + (x if self.down is None else self.down(x)))


class ResNet1d(nn.Module):
    """SE-ResNet for 12 x 1000 inputs -> token sequence [B, T, D] (T ~ 32)."""

    def __init__(self, widths=(64, 128, 192, 256), mixstyle=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, widths[0], 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers.append(nn.Sequential(Block(cin, w, 1 if i == 0 else 2), Block(w, w, 1)))
            cin = w
        self.layers = nn.ModuleList(layers)
        self.mix = MixStyle1d() if mixstyle else None
        self.dim = widths[-1]

    def forward(self, x):
        h = self.stem(x)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.mix is not None and i < 2:
                h = self.mix(h)
        return h.transpose(1, 2)  # [B, T, D]


class LinearHead(nn.Module):
    def __init__(self, dim, n_labels):
        super().__init__()
        self.fc = nn.Linear(dim, n_labels)

    def forward(self, tokens):
        return self.fc(tokens.mean(1)), None


class TextQueryHead(nn.Module):
    """Label queries derived from frozen text embeddings attend over ECG tokens.

    logit_c = s * cos(h(z_c), g(t_c)) + b, where z_c is the label-specific evidence pooled by
    cross-attention. No per-label parameters, so any described diagnosis can be scored (zero-shot).
    """

    def __init__(self, dim, text_dim, n_layers=1, heads=4, max_tokens=64, residual="none", n_labels=0):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(text_dim, dim), nn.LayerNorm(dim))
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(dim, heads, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
            for _ in range(n_layers))
        self.h = nn.Linear(dim, dim)
        self.g = nn.Linear(text_dim, dim)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))
        # Optional class-specific residual for SEEN diagnoses (zero-initialised, no RNG use, so the text part is
        # initialised exactly as without it). Unseen diagnoses (seen=0) keep the pure text-anchored logit.
        self.residual = residual
        if residual != "none":
            self.register_buffer("seen", torch.zeros(n_labels))
            self.res_b = nn.Parameter(torch.zeros(n_labels))  # per-class bias (class prior)
            if residual in ("query", "both"):  # per-class readout of the label's own cross-attended evidence
                self.res_q = nn.Parameter(torch.zeros(n_labels, dim))
            if residual in ("pool", "both"):  # per-class linear readout of mean-pooled tokens (linear-head path)
                self.res_p = nn.Parameter(torch.zeros(n_labels, dim))

    def forward(self, tokens, text, cols=None):
        B, T, _ = tokens.shape
        mem = tokens + self.pos[:, :T]
        q = self.q(text).unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, mem)
        z = F.normalize(self.h(q), dim=-1)
        t = F.normalize(self.g(text), dim=-1)
        logits = self.scale * (z * t.unsqueeze(0)).sum(-1) + self.bias
        if self.residual != "none":
            idx = slice(None) if cols is None else cols
            r = self.res_b[idx].unsqueeze(0)
            if self.residual in ("query", "both"):
                r = r + (q * self.res_q[idx].unsqueeze(0)).sum(-1)
            if self.residual in ("pool", "both"):
                r = r + tokens.mean(1) @ self.res_p[idx].T
            logits = logits + r * self.seen[idx].unsqueeze(0)
        return logits, z


class ECGModel(nn.Module):
    """head: 'linear' (mean-pool + FC), 'text' (frozen text anchors as queries),
    'query' (same decoder with learned label embeddings instead of text: ablation of text anchoring),
    'hybrid' (linear head identical to the baseline + pure text-anchored head on the same encoder; seen diagnoses
    are scored by the linear head or the mean of both probabilities, unseen diagnoses by the text head only)."""

    def __init__(self, n_labels, head="linear", text_dim=768, mixstyle=False, n_domains=0, residual="none"):
        super().__init__()
        self.enc = ResNet1d(mixstyle=mixstyle)
        d = self.enc.dim
        self.kind = head
        if head in ("linear", "hybrid"):
            self.head = LinearHead(d, n_labels)
        elif residual == "none":
            self.head = TextQueryHead(d, text_dim)
        else:
            self.head = TextQueryHead(d, text_dim, residual=residual, n_labels=n_labels)
        self.query_emb = nn.Parameter(0.02 * torch.randn(n_labels, text_dim)) if head == "query" else None
        self.dom = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_domains)) if n_domains else None
        if head == "hybrid":
            # built last and with the global RNG state restored afterwards, so the encoder / linear head / domain
            # head and every later random draw are exactly those of the linear baseline with the same seed
            with torch.random.fork_rng(devices=[]):
                self.text_head = TextQueryHead(d, text_dim)
            self.register_buffer("seen", torch.zeros(n_labels))  # set by train.py: 1 = supervised in training
            self.seen_score = "linear"
            self.text_detach = False

    def forward_both(self, x, text, cols=None, with_text=True):
        """hybrid training: (linear logits, text logits or None, pooled features)."""
        tok = self.enc(x)
        lin, _ = self.head(tok)
        lin = lin if cols is None else lin[:, cols]
        txt = None
        if with_text:
            txt, _ = self.text_head(tok.detach() if self.text_detach else tok, text if cols is None else text[cols])
        return lin, txt, tok.mean(1)

    def forward(self, x, text=None, cols=None):
        """cols: optional label indices to score (held-out labels are excluded during training)."""
        if self.kind == "hybrid":  # inference: combined logits (sigmoid of them = the reported probability)
            lin, txt, feat = self.forward_both(x, text, cols)
            seen = (self.seen if cols is None else self.seen[cols]) > 0.5
            lin, txt = lin.float(), txt.float()
            if self.seen_score == "avg":
                p = 0.5 * (torch.sigmoid(lin) + torch.sigmoid(txt))
                lin = torch.logit(p.clamp(1e-7, 1 - 1e-7))
            return torch.where(seen.unsqueeze(0), lin, txt), feat
        tok = self.enc(x)
        if self.kind == "linear":
            logits, _ = self.head(tok)
            return (logits if cols is None else logits[:, cols]), tok.mean(1)
        emb = self.query_emb if self.kind == "query" else text
        if getattr(self.head, "residual", "none") != "none":
            logits, _ = self.head(tok, emb if cols is None else emb[cols], cols)
        else:
            logits, _ = self.head(tok, emb if cols is None else emb[cols])
        return logits, tok.mean(1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None
