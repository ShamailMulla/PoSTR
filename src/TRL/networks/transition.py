"""
Transformer architecture for transition network
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..common.settings import TM_OPTIM


class EpisodeLockedDropout(nn.Module):
    """
    Dropout replacement that supports two modes:

    Unlocked (default): standard inverted dropout, identical to nn.Dropout.
    Locked: a fixed binary mask sampled once via lock() and reused every forward
            call until unlock() is called.  Used to freeze the sampled world
            hypothesis M̂ for the duration of one episode (Thompson Sampling).

    Shape-guard: if the stored mask shape does not match the input, the layer
    falls back to stochastic dropout automatically.  This means training batches
    (different N / seq_len) always get fresh random masks even when a rollout
    mask is locked.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self._mask: torch.Tensor | None = None

    def lock(self, shape: tuple, device):
        """Sample and store an inverted dropout mask for one episode."""
        scale = 1.0 / (1.0 - self.p) if self.p < 1.0 else 0.0
        self._mask = (torch.rand(shape, device=device) > self.p).float() * scale

    def unlock(self):
        """Clear the stored mask — return to stochastic training mode."""
        self._mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mask is not None and self._mask.shape == x.shape:
            return x * self._mask
        return F.dropout(x, p=self.p, training=self.training)


class LoRALinear(nn.Module):
    """
    v8: low-rank adapter around a frozen base Linear (Hu et al. 2021).
    forward(x) = base(x) + (x @ Aᵀ @ Bᵀ) · (α/r)

    B is zero-initialised, so until the adapter is enabled the wrapper is
    exactly the base layer. Pre-freeze the base trains normally and the
    adapter is inert (requires_grad False); at the freeze trigger the agent
    flips: base frozen, adapter trainable. Drift is thereby confined to a
    rank-r subspace per layer while the base dictionary is preserved.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def adapter_norm(self):
        return (self.lora_B @ self.lora_A).norm().item() * self.scaling


def inject_lora(encoder: "Encoder", rank: int, alpha: float):
    """Wrap the encoder's attention projections and FFN linears with LoRA
    adapters. Must run BEFORE the optimizer is built so adapter parameters
    are registered with it."""
    for block in encoder.layers:
        a = block.attention
        a.queries = LoRALinear(a.queries, rank, alpha)
        a.keys = LoRALinear(a.keys, rank, alpha)
        a.values = LoRALinear(a.values, rank, alpha)
        a.fully_connected_out = LoRALinear(a.fully_connected_out, rank, alpha)
        block.feed_forward[0] = LoRALinear(block.feed_forward[0], rank, alpha)
        block.feed_forward[2] = LoRALinear(block.feed_forward[2], rank, alpha)


class MultiHeadAttention(nn.Module):
    """
    Multihead attention splits an input embedding into parts for further processing
    #heads should divide embed_size equally
    For example: if embed_size is 256 then by choosing 8 heads,
                    the input embedding is divided into 256/8=32 equal parts

    # multi head attention block
    - query
    - key
    - value
    """
    def __init__(self, embed_size, heads, p_attn, device):
        super().__init__()
        self.device = device
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.p_attn = p_attn

        # BayesFormer-style per-projection dropout (sites ii) — models epistemic
        # uncertainty over attention routing (W^Q, W^K, W^V parameter uncertainty)
        self.q_dropout = EpisodeLockedDropout(p_attn)
        self.k_dropout = EpisodeLockedDropout(p_attn)
        self.v_dropout = EpisodeLockedDropout(p_attn)
        # Attention map dropout post-softmax (site iii)
        self.attn_map_dropout = EpisodeLockedDropout(p_attn)

        # creating vectors for query, key, value
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # combining all the calculated multi-headed attention to get the entire embedding
        self.fully_connected_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, queries, keys, values):
        """
        Applies EpisodeLockedDropout to Q, K, V projections and the attention map
        (BayesFormer style).  When masks are locked, each of these sites uses the
        same frozen mask throughout the episode.

        Attention(Q,K,V) = softmax(Q*K.T/sqrt(embed_size)) * V
        """
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into #heads
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        keys    = keys.reshape(N, key_len,   self.heads, self.head_dim)
        values  = values.reshape(N, value_len, self.heads, self.head_dim)

        # BayesFormer Q/K/V dropout (site ii) — locked during rollout
        queries_masked = self.q_dropout(queries)
        keys_masked    = self.k_dropout(keys)
        values_masked  = self.v_dropout(values)

        values_out  = self.values(values_masked)
        keys_out    = self.keys(keys_masked)
        queries_out = self.queries(queries_masked)

        # dot product of queries & keys
        # output shape: (N, #heads, query_len, key_len)
        queries_X_keys = torch.einsum("nqhd,nkhd->nhqk", [queries_out, keys_out])

        # attention map dropout post-softmax (site iii) — locked during rollout
        queries_X_keys_scaled = self.attn_map_dropout(
            torch.softmax(queries_X_keys / self.embed_size ** 0.5, dim=3)
        )

        # expose the true attention map (N, heads, query_len, key_len) for
        # post-hoc interpretability; behaviour-preserving (attribute only).
        self.attention_map = queries_X_keys_scaled.detach()

        # output shape: (N, query_len, #heads, head_dim)
        attention = torch.einsum("nhql,nlhd->nqhd", [queries_X_keys_scaled, values_out])
        attention = attention.reshape(N, query_len, self.heads * self.head_dim)

        out = self.fully_connected_out(attention)
        return out, attention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, p_attn, forward_expansion, device):
        super().__init__()

        self.attention = MultiHeadAttention(embed_size, heads, p_attn, device)
        self.norm = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        # FFN block skip-connection dropout (site iv) — locked during rollout
        self.dropout = EpisodeLockedDropout(p_attn)

    def forward(self, query, key, value):
        """
        Attention + skip + norm + FFN + skip + norm.
        Both skip-connection dropout calls share the same EpisodeLockedDropout
        instance (same shape at both sites).
        """
        attention, self.attention_scores = self.attention(query, key, value)

        x = self.dropout(self.norm(attention + query))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm(forward + x))

        return out, self.attention_scores


class Encoder(nn.Module):
    """
    Input sequence is embedded in model embedding dimension.
    Contains multi-head self attention blocks to encode embedding.
    ENCODER BLOCK(s):
        1. ATTENTION BLOCK (multi head)
        2. SKIP CONNECTION & NORMALISE
        3. FEED FWD N/W
        4. SKIP CONNECTION & NORMALISE
    """
    def __init__(self, n_state: int, n_action: int, embed_size: int,
                 num_layers: int, heads: int, forward_expansion: int,
                 p_seq: float, p_attn: float, max_timesteps: int, device):
        super().__init__()
        self.device = device
        self.embed_size = embed_size

        self.norm = nn.LayerNorm(embed_size)
        self.embed_state     = nn.Linear(n_state, embed_size).to(device)
        self.embed_action    = nn.Embedding(n_action, embed_size).to(device)
        self.embed_timestep  = nn.Embedding(max_timesteps, embed_size).to(device)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, p_attn, forward_expansion, device)
            for _ in range(num_layers)
        ])

        # Input embedding dropout (site i) — models observation noise; locked during rollout
        self.state_dropout  = EpisodeLockedDropout(p_seq)
        self.pos_dropout    = EpisodeLockedDropout(p_seq)
        self.action_dropout = EpisodeLockedDropout(p_seq)

        self.to(device)

    def forward(self, timesteps: torch.Tensor, states: torch.Tensor, actions: torch.Tensor):
        """
        Encode the context sequence.  All EpisodeLockedDropout layers use their
        stored masks when locked (episode rollout) or fresh random masks otherwise
        (BayesFormer training).
        """
        self.attention_scores = []
        N, seq_len, _ = states.shape

        embedded_states   = self.embed_state(states)
        embedded_timestep = self.embed_timestep(timesteps)

        # site i — input embedding dropout (locked during rollout)
        states_masked    = self.state_dropout(embedded_states)
        timesteps_masked = self.pos_dropout(embedded_timestep)

        embedded_state = self.norm(
            states_masked * self.embed_size ** 0.5 + timesteps_masked
        )

        # action embedding dropout (site i, action branch) — locked during rollout
        embedded_action = self.action_dropout(
            self.norm(
                self.embed_action(actions) * (self.embed_size ** 0.5) + embedded_timestep
            )
        )

        # interleave states and actions: (s0,a0, s1,a1, ...) → (N, 2*seq_len, embed)
        out = torch.stack(
            (embedded_state, embedded_action), dim=2
        ).view(N, -1, self.embed_size)

        for transformer_layer in self.layers:
            out, attention_scores = transformer_layer(out, out, out)
            self.attention_scores.append(attention_scores.clone())

        return out


class Network(nn.Module):
    def __init__(self, config: dict, states: int, actions: list,
                 p_seq=0.2, p_attn=0.05, max_steps=1000,
                 device=torch.device):
        super().__init__()

        self.context_length = config["context_length"]
        self.latent_dim = self.context_length + config["hidden_dim"]
        self.num_actions = len(actions)
        self.max_T = max_steps
        self.states = states
        self.obs_type = config.get("obs_type", "grid")
        self.to(device)
        self.loss = 0

        self.encoder = Encoder(
            states, self.num_actions,
            config["hidden_dim"], config['num_encoder_layers'], config['heads'],
            config['forward_expansion'], p_seq, p_attn, max_steps, device
        )

        # v8: LoRA adapters (inert until the freeze trigger enables them).
        # Injected before the optimizer is built so their parameters register.
        self.lora_rank = int(config.get("lora_rank", 0))
        if self.lora_rank > 0:
            inject_lora(self.encoder, self.lora_rank, float(config.get("lora_alpha", 2 * self.lora_rank)))
            self.encoder.to(device)

        ### prediction heads
        self.predict_rtg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config["hidden_dim"] * 2 * config['context_length'], config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], 1)
        ).to(device)
        self.predict_state = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config["hidden_dim"] * 2 * config['context_length'], config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], states),
        ).to(device)

        # v9 fix: historically the optimizer was created BEFORE the prediction
        # heads, so predict_rtg/predict_state were never trained (frozen random
        # projections) in every arm up to v8. train_heads=True includes them;
        # the legacy encoder-only parameter set is kept as the default so older
        # configs reproduce exactly. (Module construction order is unchanged,
        # so RNG streams and initial weights match legacy runs either way.)
        params = self.parameters() if config.get("train_heads", False) else self.encoder.parameters()
        self.optimizer = TM_OPTIM(params, lr=config["learning_rate"])

    # ------------------------------------------------------------------
    # Episode-mask management (Problem #1 & #2 fix)
    # ------------------------------------------------------------------

    def lock_all_dropouts(self, N: int, seq_len: int, device):
        """
        Sample and lock every BayesFormer dropout site for one episode.

        N       = num_actions  (rollout batch dimension)
        seq_len = context_length  (before state-action interleaving)
        """
        embed = self.encoder.embed_size
        eff   = 2 * seq_len          # after state-action interleaving in Encoder
        enc   = self.encoder

        # site i: input embedding dropouts (pre-interleave seq_len)
        enc.state_dropout.lock( (N, seq_len, embed), device)
        enc.pos_dropout.lock(   (N, seq_len, embed), device)
        enc.action_dropout.lock((N, seq_len, embed), device)

        for block in enc.layers:
            a = block.attention
            # site ii: Q/K/V projection dropouts (post-interleave: eff = 2*seq_len)
            a.q_dropout.lock(        (N, eff, a.heads, a.head_dim), device)
            a.k_dropout.lock(        (N, eff, a.heads, a.head_dim), device)
            a.v_dropout.lock(        (N, eff, a.heads, a.head_dim), device)
            # site iii: attention map dropout
            a.attn_map_dropout.lock( (N, a.heads, eff, eff),        device)
            # site iv: FFN skip-connection dropout
            block.dropout.lock(      (N, eff, embed),               device)

    def unlock_all_dropouts(self):
        """Clear all locked masks — return to stochastic BayesFormer training mode."""
        enc = self.encoder
        enc.state_dropout.unlock()
        enc.pos_dropout.unlock()
        enc.action_dropout.unlock()
        for block in enc.layers:
            block.attention.q_dropout.unlock()
            block.attention.k_dropout.unlock()
            block.attention.v_dropout.unlock()
            block.attention.attn_map_dropout.unlock()
            block.dropout.unlock()

    # ------------------------------------------------------------------

    def encode(self, timesteps, state, action):
        return self.encoder(timesteps, state, action)

    def forward(self, step: torch.Tensor, observations: torch.Tensor, actions: torch.Tensor):
        encoded_output = self.encode(step, observations, actions)

        return_preds     = self.predict_rtg(encoded_output)
        next_state_preds = self.predict_state(encoded_output)
        attention_scores = torch.stack(self.encoder.attention_scores)
        return next_state_preds, return_preds, attention_scores

    def predict(self, step: torch.Tensor, observations: torch.Tensor, actions: torch.Tensor):
        with torch.no_grad():
            encoded_output   = self.encode(step, observations, actions)
            return_preds     = self.predict_rtg(encoded_output)
            next_state_preds = self.predict_state(encoded_output)
        return next_state_preds, return_preds

    def features(self, step: torch.Tensor, observations: torch.Tensor, actions: torch.Tensor):
        """
        Penultimate features Φ for the v3 neural-linear head: the POST-nonlinearity
        representation of predict_state (Flatten -> Linear -> ReLU), i.e.
        predict_state[:-1] applied to the encoder output. Shape (N, hidden_dim).

        This is the PSDRL-faithful locus (BLR replaces the final linear layer, so
        the BLR posterior mean can match the trained head's accuracy) and is 4x
        smaller than the raw flattened encoder output — better-conditioned and
        cheaper for the closed-form posterior. The deterministic predict_state /
        predict_rtg heads remain for gradient-training these features; the
        neural-linear head fits a Bayesian linear layer on Φ for the posterior
        over world models (replacing dropout-as-posterior).

        Note: reward is regressed from the state head's penultimate — valid on
        DeepSea (reward is a function of the next state); a shared feature head
        is the generalisation if needed.
        """
        with torch.no_grad():
            encoded_output = self.encode(step, observations, actions)
            phi = self.predict_state[:-1](encoded_output)   # Flatten -> Linear -> ReLU
        return phi