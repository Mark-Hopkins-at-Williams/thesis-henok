import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def forward(self, e, f, e_attn_mask, f_attn_mask):
        Q, K, V = e, f, f
        scores = (
            Q @ K.transpose(-2, -1)
        ) * self.k  # TODO: what should this multiplier be?

        scores = f_attn_mask.unsqueeze(1) * scores
        scores = e_attn_mask.unsqueeze(2) * scores
        print(scores)
        exit()
        scores = (-100 * (e_attn_mask == 0).int()).unsqueeze(1) + scores
        scores = (-100 * (f_attn_mask == 0).int()).unsqueeze(2) + scores
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights


if __name__ == "__main__":
    batch_size, seq_len, embed_dim = 2, 3, 5

    encoder_states_sent = torch.tensor(
        [
            [
                [0.2, 0.4, 0.6, -0.2, -1.0],  # d_model = 5
                [-0.3, -1.4, 1.2, -0.1, 0.7],
                [0.4, 0.2, -0.6, 0.2, 0.5],
            ],
            [
                [0.1, 0.2, 0.3, -0.5, -1.2],  # d_model = 5
                [-0.3, -1.4, 1.2, -0.1, 0.7],
                [0.4, 0.2, -0.6, 0.2, 0.5],
            ],  # each sent has 3 token embeddings
        ]  # batch of 2 sents
    )

    encoder_states_goal = torch.tensor(
        [
            [
                [0.2, 0.4, 0.6, -0.2, -1.0],  # d_model = 5
                [-0.3, -1.4, 1.2, -0.1, 0.7],
                [0.4, 0.2, -0.4, 0.21, 0.5],
                [0.4, 0.2, -0.6, 0.2, 0.5],
            ],
            [
                [0.1, 0.2, 0.3, -0.5, -1.2],  # d_model = 5
                [-0.3, -1.4, 1.2, -0.1, 0.7],
                [0.4, 0.2, -0.6, 0.2, 0.5],
                [0.4, 0.2, -0.6, 0.2, 0.5],
            ],  # each sent has 4 token embeddings
        ]  # batch of 2 sents
    )

    sigma = 0.04
    # noise = sigma * torch.randn_like(encoder_states_goal)
    # encoder_states_corrupted = (encoder_states_gold + noise).flip(0)
    print("sent:")
    print(encoder_states_sent)
    print("goal:")
    print(encoder_states_goal)

    attn = SimpleAttention()
    sent_attn_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    goal_attn_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    out, weights = attn(
        encoder_states_sent, encoder_states_goal, sent_attn_mask, goal_attn_mask
    )

    print("out:")
    print(out)
    print("weights:")
    print(weights)

    token_scores = 1 - F.cosine_similarity(
        encoder_states_goal, out, dim=-1
    )  # [seq_len]
    loss = token_scores.mean()
    print(f"loss: {loss}")
    exit()

    print("Output shape:", out.shape)  # (2, 5, 16)
    print(out)
    # print("Attention weights shape:", weights.shape)  # (2, 5, 5)

    print(token_scores)
