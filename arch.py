# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from vit_pytorch import ViT
import matplotlib.pyplot as plt


# %%
class PVT(nn.Module):
    """Packet-based Vision Transformer"""

    def __init__(self, in_channels=1, image_size=64, patch_size=16, max_packet_len=1502, out_dim=256, depth=2, heads=12, mlp_dim=2048):
        super().__init__()
        self.vit = ViT(
            channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            dim=out_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            num_classes=1,   # dummy; classification head not used
        )
        self.packet_len_embedding = nn.Embedding(max_packet_len, out_dim)

    def forward(self, x, packet_lens):
        # x:            (B, C, H, W)
        # packet_lens:  (B,) — ints in [0, max_packet_len)
        B = x.shape[0]

        # 1. Patch embedding
        x = self.vit.to_patch_embedding(x)               # (B, num_patches, dim)

        # 2. Look up and add custom embedding per patch
        custom = self.packet_len_embedding(packet_lens)  # (B, num_patches, dim)
        x = x + custom

        # 3. Prepend CLS token
        cls = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                   # (B, num_patches+1, dim)

        # 4. Add positional embedding
        x += self.vit.pos_embedding

        # paper does NOT use dropout!
        #x = self.vit.dropout(x)
        x = self.vit.transformer(x)
        x = self.vit.to_latent(x[:, 0])                  # CLS token output

        return x


# %%
class ResAtConv(nn.Module):
    """Residual Attention Convolutional Block"""

    def __init__(self, in_channels=1, conv_channels=8, kernel_size=4, stride=2, padding=0, fc_dim=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(conv_channels)

        if fc_dim is None:
            fc_dim = conv_channels

        self.gap_linear = nn.Linear(conv_channels, fc_dim)
        self.gmp_linear = nn.Linear(conv_channels, fc_dim)
        self.compressed_linear = nn.Linear(fc_dim, conv_channels)

        init.xavier_uniform_(self.gap_linear.weight, gain=init.calculate_gain('tanh'))
        init.xavier_uniform_(self.gmp_linear.weight, gain=init.calculate_gain('tanh'))
        init.xavier_uniform_(self.compressed_linear.weight, gain=init.calculate_gain('sigmoid'))

        self.conv_1x1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=1)

        self.feats_dim = lambda patch_size: conv_channels * ((patch_size + 2 * padding - kernel_size) // stride + 1) ** 2

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.bn(self.conv(x))
        gap = F.adaptive_avg_pool2d(feats, 1).flatten(start_dim=1)  # (B, conv_channels)
        gmp = F.adaptive_max_pool2d(feats, 1).flatten(start_dim=1)
        compressed = torch.tanh(self.gap_linear(gap)) + torch.tanh(self.gmp_linear(gmp))
        attention = torch.sigmoid(self.compressed_linear(compressed))
        attention = attention.unsqueeze(-1).unsqueeze(-1)           # (B, conv_channels, 1, 1)
        feats = feats + attention * feats
        feats = self.conv_1x1(feats).flatten(start_dim=1)           # (B, feats_dim)
        return feats


# %%
class STFE(nn.Module):
    """Spatial-Temporal Feature Extractor"""

    def __init__(
        self,
        in_channels:    int = 1,
        patch_size:     int = 16,
        out_dim:        int = 256,
        rac_kwargs:     dict | None = None,
        lstm_hidden:    int | None = None,
        lstm_layers:    int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size

        rac_kwargs = rac_kwargs or {}
        self.rac = ResAtConv(in_channels=in_channels, **rac_kwargs)

        # Derive the flat feature size that ResAtConv produces per patch
        feat_dim = self.rac.feats_dim(patch_size)

        if lstm_hidden is None:
            lstm_hidden = feat_dim

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Concat of forward + backward last hidden → out_dim
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, out_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        P = self.patch_size

        # 1. Extract non-overlapping patches → (B, C, nH, nW, P, P)
        patches = x.unfold(2, P, P).unfold(3, P, P)
        _, _, nH, nW, _, _ = patches.shape
        N = nH * nW

        # 2. Flatten batch × patches for ResAtConv → (B*N, C, P, P)
        patches_flat = patches.contiguous().view(B * N, C, P, P)

        # 3. ResAtConv per patch → (B*N, feat_dim)
        feats_flat = self.rac(patches_flat)

        # 4. Restore patch sequence → (B, N, feat_dim)
        feats = feats_flat.view(B, N, -1)

        # 5. Bi-LSTM over the patch sequence
        #    output: (B, N, 2*lstm_hidden)
        #    hn:     (num_layers*2, B, lstm_hidden), unaffected by batch_first=True!
        _, (hn, _) = self.lstm(feats)

        # hn[-2] = last layer forward final state
        # hn[-1] = last layer backward final state
        fwd = hn[-2]   # (B, lstm_hidden)
        bwd = hn[-1]   # (B, lstm_hidden)
        combined = torch.cat([fwd, bwd], dim=-1)  # (B, lstm_hidden*2)

        # 6. Project to output dim
        out = self.head(combined)   # (B, out_dim)
        return out


# %%
class ATVITSC(nn.Module):
    """Attention-based Vision Transformer and Spatiotemporal for Traffic Classification"""

    def __init__(
        self,
        # ── sub-module i/o ──
        channels:           int = 1,
        image_size:         int = 64,
        patch_size:         int = 16,
        max_packet_len:     int = 1502,
        shared_feat_dim:    int = 256,
        num_classes:        int = 2,    # C:  number of traffic classes
        # ── dynamic weighting ──
        dw_hidden:   int = 128,         # h:  dimension of intermediate vector z
        temperature: float = 500.0,     # τ:  softmax temperature
        # ── classifier head ──
        cls_hidden:  int = 128,         # hidden dim between the two FC layers
        # ── sub-module configs ──
        pvt_kwargs:  dict | None = None,
        stfe_kwargs: dict | None = None,
    ):
        super().__init__()

        self.temperature = temperature

        # ── Sub-modules ──────────────────────────────────────────────────────
        pvt_kwargs  = pvt_kwargs  or {}
        stfe_kwargs = stfe_kwargs or {}

        self.pvt  = PVT(
            in_channels=channels,
            image_size=image_size,
            patch_size=patch_size,
            max_packet_len=max_packet_len,
            out_dim=shared_feat_dim,
            **pvt_kwargs
        )
        self.stfe = STFE(
            in_channels=channels,
            patch_size=patch_size,
            out_dim=shared_feat_dim,
            **stfe_kwargs
        )

        # ── Dynamic weighting network ─────────────────────────────────────
        # Step 2: [F_global; F_st] ∈ R^{2d}  →  z ∈ R^h  (tanh)
        self.W_gs = nn.Linear(shared_feat_dim * 2, dw_hidden)

        # Step 3: z ∈ R^h  →  ξ ∈ R^2
        self.W_z  = nn.Linear(dw_hidden, 2)

        # ── Classifier head ───────────────────────────────────────────────
        # Step 6: v ∈ R^d  →  R^cls_hidden  →  R^C
        self.classifier = nn.Sequential(
            nn.Linear(shared_feat_dim, cls_hidden),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(cls_hidden, num_classes),
        )

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, packet_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x           : (B, C, H, W)      — raw packet image(s)
            packet_lens : (B, num_patches)  — length of each packet in bytes

        Returns:
            y : (B, C)  — log-softmax probability vector over traffic classes
        """
        # ── Step 0: extract features ──────────────────────────────────────
        f_global = self.pvt(x, packet_lens)                 # (B, d)
        f_st     = self.stfe(x)                             # (B, d)

        # ── Step 1: concatenate ───────────────────────────────────────────
        f_gs = torch.cat([f_global, f_st], dim=-1)          # (B, 2d)

        # ── Step 2: intermediate tanh projection ──────────────────────────
        z = torch.tanh(self.W_gs(f_gs))                     # (B, h)

        # ── Step 3: score projection ──────────────────────────────────────
        xi = self.W_z(z)                                    # (B, 2)

        # ── Step 4: temperature-scaled softmax weights ────────────────────
        alpha = F.softmax(xi / self.temperature, dim=-1)    # (B, 2)
        alpha_g = alpha[:, :1]                              # (B, 1)  — broadcast-friendly
        alpha_s = alpha[:, 1:]                              # (B, 1)

        # ── Step 5: fused feature vector ──────────────────────────────────
        v = alpha_g * f_global + alpha_s * f_st             # (B, d)

        # ── Step 6: classify ──────────────────────────────────────────────
        logits = self.classifier(v)                         # (B, C)
        return logits


# %%
# ─────────────────────────── Tests ───────────────────────────
if __name__ == '__main__':
    B          = 2
    img_size   = 64
    patch_size = 16
    num_patches   = (img_size // patch_size) ** 2

    x_img         = torch.randn(B, 1, img_size, img_size)
    patch_indices = torch.randint(0, 1502, (B, num_patches))

# %%
# Plot input image with patches labeled
if __name__ == '__main__':
    img_np = x_img[0].squeeze().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np, cmap='gray')
    for y_pos in range(0, img_size, patch_size):
        for x_pos in range(0, img_size, patch_size):
            rect = plt.Rectangle((x_pos - 0.5, y_pos - 0.5), patch_size, patch_size,
                                  edgecolor='r', facecolor='none', lw=0.5)
            ax.add_patch(rect)
            patch_idx = (y_pos // patch_size) * (img_size // patch_size) + (x_pos // patch_size)
            ax.text(x_pos + patch_size / 2 - 0.5, y_pos + patch_size / 2 - 0.5,
                    str(patch_idx), color='white', fontsize=6, ha='center', va='center')
    plt.title(f"Random image — {patch_size}×{patch_size} patches")
    plt.axis('off')
    plt.show()

# %%
# Test PVT
if __name__ == '__main__':
    pvt = PVT()
    pvt_out = pvt(x_img, patch_indices)
    print(f"PVT output shape: {pvt_out.shape}")

# %%
# Test ResAtConv patch-by-patch (manual, to verify patch extraction)
if __name__ == '__main__':
    rac = ResAtConv()

    patches = x_img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    B_, C_, nH, nW, pH, pW = patches.shape
    N = nH * nW
    patches_flat = patches.contiguous().view(B_ * N, C_, pH, pW)
    rac_out = rac(patches_flat).view(B_, N, -1)
    print(f"ResAtConv patch output shape: {rac_out.shape}")

    # Confirm first patch matches top-left region of the image
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    orig_np = x_img[0, 0].numpy()
    orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min())
    axes[0].imshow(orig_np, cmap='gray')
    axes[0].add_patch(plt.Rectangle((-0.5, -0.5), patch_size, patch_size,
                                     edgecolor='red', facecolor='none', lw=2))
    axes[0].set_title("Original (patch 0 highlighted)")
    axes[0].axis('off')

    patch_np = patches_flat[0, 0].numpy()
    patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min())
    axes[1].imshow(patch_np, cmap='gray')
    axes[1].set_title(f"patches_flat[0]  shape={tuple(patches_flat[0].shape)}")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

# %%
# Test STFE
if __name__ == '__main__':
    stfe = STFE()
    stfe_out = stfe(x_img)
    print(f"STFE output shape: {stfe_out.shape}")

# %%
# Test ATVITSC
if __name__ == '__main__':
    C_classes = 8
    model = ATVITSC(num_classes=C_classes)
    y = model(x_img, patch_indices)

    assert y.shape == (B, C_classes), f"Unexpected output shape: {y.shape}"
    print(f"Input shape     : {tuple(x_img.shape)}")
    print(f"Output shape    : {tuple(y.shape)}")
    print(f"Predicted class : {y.argmax(dim=-1).tolist()}")
    print("ATVITSC test passed ✓")

# %%
