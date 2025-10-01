#!POPCORN leaderboard trimul
import torch
import torch.nn.functional as F
from task import input_t, output_t
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

def compiledtrimul(
    x: torch.Tensor,
    mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    w_concat: torch.Tensor,
    to_out_norm_weight: torch.Tensor,
    to_out_norm_bias: torch.Tensor,
    to_out_weight: torch.Tensor,
    h: int
) -> torch.Tensor:
    """
    A barebones, compiled PyTorch function for the TriMul logic.
    """
    bs, s1, s2, d = x.shape

    # Initial LayerNorm
    x_norm = F.layer_norm(x, (d,), norm_weight, norm_bias).view((bs * s1 * s2, d)).to(torch.float16)
    # Single large matmul: [M, d] @ [d, 5h] = [M, 5h]
    all_projections = torch.mm(x_norm, w_concat)

    # Split back into individual projections
    left, right, lg, rg, og = all_projections.chunk(5, dim=1)

    # Apply mask and gates
    mask_expanded = mask.expand(-1, -1, -1, h).reshape(-1, h)
    left = left * mask_expanded * torch.sigmoid(lg)
    right = right * mask_expanded * torch.sigmoid(rg)

    # Reshape for einsum
    left = left.view(bs, s1, s2, h).permute(0,3,1,2)
    right = right.view(bs, s1, s2, h).permute(0,3,1,2)
    out_p = torch.matmul(left.to(torch.float16), right.to(torch.float16).transpose(-1, -2))
    out_einsum_flat = out_p.permute(0,2,3,1).reshape(bs * s1 * s1, h)

    # Apply layer norm and final gating
    normed = F.layer_norm(out_einsum_flat, (h,), to_out_norm_weight, to_out_norm_bias).to(torch.float16)
    gated = normed * torch.sigmoid(og)

    # Final projection
    final_out_flat = gated @ to_out_weight.t()
    final_out = final_out_flat.view(bs, s1, s2, d)

    return final_out
def custom_kernel(data: input_t) -> output_t:
    """
    Refactored implementation of TriMul using a barebones compiled PyTorch function.

    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
    """
    input_tensor, mask, weights, config = data
    w_concat = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight']
    ], dim=0).t().contiguous().to(torch.float16)
    # Call the compiled function with prepared weights
    output = compiledtrimul(
        x=input_tensor.to(torch.float32),
        mask=mask.unsqueeze(-1),
        norm_weight=weights['norm.weight'].to(torch.float32),
        norm_bias=weights['norm.bias'].to(torch.float32),
        w_concat=w_concat,
        to_out_norm_weight=weights['to_out_norm.weight'].to(torch.float32),
        to_out_norm_bias=weights['to_out_norm.bias'].to(torch.float32),
        to_out_weight=weights['to_out.weight'].to(torch.float16),
        h=config["hidden_dim"]
    )
    return output
