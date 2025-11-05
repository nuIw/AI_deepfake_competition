class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        x = F.normalize(x)
        W = F.normalize(self.weight)
        logits = x @ W.t()  # (B, 2)

        # ✅ 추론일 때 (labels=None)
        if labels is None:
            return logits * self.s

        # ✅ 학습일 때 (margin 적용)
        logits = torch.clamp(logits, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(logits)
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = logits * (1 - one_hot) + target_logits * one_hot
        return output * self.s


class DeepfakeCLIP_Arc(nn.Module):
    def __init__(self):
        super().__init__()
        base = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.backbone = base.vision_model   # ✅ 여기서 vision_model만 가져옴
        dim = self.backbone.config.hidden_size

        self.norm = nn.LayerNorm(dim)
        self.arc = ArcMarginProduct(dim, 2)

        # freeze all
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ✅ 마지막 2개 block만 fine-tune
        for p in self.backbone.encoder.layers[-2:].parameters():
            p.requires_grad = True

    def forward(self, x, labels=None):
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        h = out.hidden_states[-1]  # (B, seq, dim)
        h = h[:, 1:, :].mean(dim=1)  # Patch Mean
        h = self.norm(h)

        return self.arc(h, labels)
