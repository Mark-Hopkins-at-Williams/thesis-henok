import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def forward(self, e, f):
        Q, K, V = e, f, f
        scores = (
            Q @ K.transpose(-2, -1)
        ) * self.k  # TODO: what should this multiplier be?
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights


if __name__ == "__main__":
    batch_size, seq_len, embed_dim = 2, 3, 5

    encoder_states_gold = torch.tensor(
        [
            [0.2, 0.4, 0.6, -0.2, -1.0],  # d_model = 5
            [-0.3, -1.4, 1.2, -0.1, 0.7],
            [0.4, 0.2, -0.6, 0.2, 0.5],
        ]  # 3 token embeddings
    )

    sigma = 0.04
    noise = sigma * torch.randn_like(encoder_states_gold)
    encoder_states_corrupted = (encoder_states_gold + noise).flip(0)
    print("gold:")
    print(encoder_states_gold)
    print("corrupted:")
    print(encoder_states_corrupted)

    attn = SimpleAttention()
    out, weights = attn(encoder_states_corrupted, encoder_states_gold)

    print("out:")
    print(out)
    print("weights:")
    print(weights)

    token_scores = 1 - F.cosine_similarity(
        encoder_states_corrupted, out, dim=-1
    )  # [seq_len]
    loss = token_scores.mean()
    print(f"loss: {loss}")
    exit()

    print("Output shape:", out.shape)  # (2, 5, 16)
    print(out)
    # print("Attention weights shape:", weights.shape)  # (2, 5, 5)

    print(token_scores)
