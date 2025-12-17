import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class PhikonSegmentationModel(nn.Module):
    """
    Phikon-v2 based segmentation model.
    Uses Phikon-v2 as encoder and a simple decoder for pixel-wise predictions.
    """
    
    def __init__(
        self,
        classes: int,
        patch_size: int = 16,
        embed_dim: int = 1024,
        decoder_channels: list = [512, 256, 128, 64],
        pretrained_name: str = "owkin/phikon-v2",
        freeze_encoder: bool = True,
    ):
        """
        Args:
            classes: Number of segmentation classes
            patch_size: Size of patches (typically 16 for Phikon)
            embed_dim: Embedding dimension (1024 for Phikon-v2)
            decoder_channels: Channel sizes for decoder layers
            pretrained_name: HuggingFace model name
            freeze_encoder: Whether to freeze the Phikon encoder
        """
        super().__init__()
        
        # Load Phikon-v2 encoder
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        self.processor = AutoImageProcessor.from_pretrained(pretrained_name)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels
        self.freeze_encoder = freeze_encoder
        self.num_classes = classes
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Upsample from patch features to pixel features
        # Input: (B, 1024, H/16, W/16) -> Output: (B, num_classes, H, W)
        decoder_layers = []
        in_channels = self.embed_dim
        
        for out_channels in self.decoder_channels:
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Final classification head
        self.segmentation_head = nn.Conv2d(
            self.decoder_channels[-1], self.num_classes, kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W), values in [0, 1]
        
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Normalize image for Phikon (expects ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_normalized = (x - mean) / std
        
        # Extract features from Phikon encoder
        with torch.set_grad_enabled(self.training and not self.freeze_encoder):
            outputs = self.encoder(pixel_values=x_normalized)
            features = outputs.last_hidden_state
        
        # Remove CLS token
        patch_features = features[:, 1:, :]
        
        # Reshape to 2D feature map: (B, 1024, H/16, W/16)
        patch_features = patch_features.transpose(1, 2)
        patch_features = patch_features.reshape(
            B, self.embed_dim, num_patches_h, num_patches_w
        )
        
        # Decode to full resolution
        decoded = self.decoder(patch_features)
        
        # Get segmentation logits
        logits = self.segmentation_head(decoded)  # (B, num_classes, H, W)
        
        return logits