import torch
import torch.nn as nn

class QuantLinearPyTorch(nn.Module):
    def __init__(self, in_features, out_features, qweight: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        assert qweight.dtype == torch.int8
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16))
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        
        qweight_float = self.qweight.to(torch.float16) * self.scales.unsqueeze(1)# Dequantize the weights before matmul
        qweight_float = qweight_float.to(torch.float16)
        out = torch.matmul(x, qweight_float.t())  #B x K * K x OC => B x OC

        if self.bias is not None:
            out += self.bias.unsqueeze(0)  #Adding bias

        return out.to(orig_dtype) if orig_dtype == torch.float16 else out # Return output in og dtype

def quantize_int8_per_channel(W: torch.Tensor):
    Wf = W.to(torch.float16)
    max_vals = Wf.abs().amax(dim=1)
    scales = (max_vals / 127.0).clamp(min=1e-8).to(torch.float16)
    q = torch.round(Wf / scales.unsqueeze(1).to(Wf.dtype)).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scales.contiguous()

def replace_linear_with_pytorch(module: nn.Module, prefix="", exclude_prefixes=None):
    if exclude_prefixes is None:
        exclude_prefixes = ["lm_head", "model.embed_tokens", "model.visual", "model.language_model.embed_tokens"]
    for name, child in list(module.named_children()):
        full = prefix + "." + name if prefix else name
        if any(full.startswith(e) for e in exclude_prefixes):
            continue
        if isinstance(child, nn.Linear):
            W = child.weight.data
            b = child.bias.data if child.bias is not None else None
            q, scales = quantize_int8_per_channel(W)
            q, scales = q.cuda(), scales.cuda()
            new = QuantLinearPyTorch(child.in_features, child.out_features, q, scales,
                                     b.to(torch.float32).cuda() if b is not None else None)
            setattr(module, name, new)
        else:
            replace_linear_with_pytorch(child, full, exclude_prefixes)
    return module
