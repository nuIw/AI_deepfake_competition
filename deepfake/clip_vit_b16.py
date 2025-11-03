class DeepfakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,2)
        )
    def forward(self, x):
        x = self.backbone(pixel_values=x).pooler_output
        return self.fc(x)
