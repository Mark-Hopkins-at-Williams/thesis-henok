import torch
import torch.nn.functional as F


def compute_loss(batch, temperature=0.1):
    sents, lang, goal_encodings, goal_attn_mask = batch

    sent_attn_mask = sents["attention_mask"].to(encoder.device)
    sents = {k: v.to(encoder.device) for k, v in sents.items()}
    goal_encodings = goal_encodings.to(encoder.device)
    goal_attn_mask = goal_attn_mask.to(encoder.device)

    sent_encodings = encoder(**sents).last_hidden_state

    # R(xi): mean-pooled trainable encoder output on source (lang1)
    src_vecs = mean_pool(sent_encodings, sent_attn_mask)  # (B, H)

    # R(xj) / R(yj): mean-pooled static encoder output on goal (lang2)
    tgt_vecs = mean_pool(goal_encodings, goal_attn_mask)  # (B, H)

    # cosine similarity
    src_vecs = F.normalize(src_vecs, dim=-1)
    tgt_vecs = F.normalize(tgt_vecs, dim=-1)

    # sim[i, j] = cos(R(xi), R(tgt_j)) / τ
    sim = torch.matmul(src_vecs, tgt_vecs.T) / temperature  # (B, B)

    B = sim.size(0)

    # sim+: positive pair similarity — diagonal (xi matched with its correct translation xj)
    numerator = torch.diagonal(sim)  # (B,)

    # sim-: negatives only — mask out the diagonal so yj ≠ xj
    mask = torch.eye(B, dtype=torch.bool, device=sim.device)
    sim_neg = sim.masked_fill(mask, float('-inf'))

    # log Σ_{yj} exp(sim-(R(xi), R(yj)) / τ)
    log_denominator = torch.logsumexp(sim_neg, dim=1)  # (B,)

    # -Σ_{xi,xj ∈ D} log [ exp(sim+/τ) / Σ_{yj} exp(sim-/τ) ]
    loss = (-numerator + log_denominator).mean()
    return loss
