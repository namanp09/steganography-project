"""
Modern Loss Functions for Steganography Training.

Combines multiple loss terms:
- Image quality: MSE + MS-SSIM + LPIPS perceptual
- Message accuracy: BCE for bit recovery
- Adversarial: WGAN-GP for imperceptibility
- Frequency: DCT-domain loss for anti-steganalysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False


class MessageLoss(nn.Module):
    """Binary cross-entropy loss for message bit recovery."""

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(predicted, target)


class ImageQualityLoss(nn.Module):
    """
    Combined image quality loss:
    L = λ1 * MSE + λ2 * (1 - MS-SSIM) + λ3 * LPIPS
    """

    def __init__(self, lambda_mse=1.0, lambda_msssim=0.5, lambda_lpips=0.5):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_msssim = lambda_msssim
        self.lambda_lpips = lambda_lpips

        if HAS_LPIPS:
            self.lpips_fn = lpips.LPIPS(net="alex", verbose=False)
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        else:
            self.lpips_fn = None

    def forward(self, stego: torch.Tensor, cover: torch.Tensor) -> dict:
        losses = {}

        # MSE
        losses["mse"] = F.mse_loss(stego, cover)

        # MS-SSIM (requires images >= 160x160)
        if HAS_MSSSIM and stego.shape[-1] >= 160:
            msssim_val = ms_ssim(stego, cover, data_range=1.0, size_average=True)
            losses["ms_ssim"] = 1 - msssim_val
        else:
            losses["ms_ssim"] = torch.tensor(0.0, device=stego.device)

        # LPIPS perceptual loss
        if self.lpips_fn is not None:
            # LPIPS expects images in [-1, 1]
            lpips_val = self.lpips_fn(stego * 2 - 1, cover * 2 - 1).mean()
            losses["lpips"] = lpips_val
        else:
            losses["lpips"] = torch.tensor(0.0, device=stego.device)

        # Combined
        total = (
            self.lambda_mse * losses["mse"]
            + self.lambda_msssim * losses["ms_ssim"]
            + self.lambda_lpips * losses["lpips"]
        )
        losses["total_image"] = total
        return losses


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss — minimizes spectral differences.
    Helps evade frequency-based steganalysis detectors.
    """

    def forward(self, stego: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        # 2D FFT
        stego_fft = torch.fft.fft2(stego)
        cover_fft = torch.fft.fft2(cover)

        # Compare magnitude spectra
        stego_mag = torch.abs(stego_fft)
        cover_mag = torch.abs(cover_fft)

        return F.mse_loss(stego_mag, cover_mag)


class WGANGPLoss:
    """Wasserstein GAN with Gradient Penalty loss computation."""

    @staticmethod
    def generator_loss(discriminator_output: torch.Tensor) -> torch.Tensor:
        """Generator wants discriminator to output high values for stego."""
        return -discriminator_output.mean()

    @staticmethod
    def discriminator_loss(
        real_output: torch.Tensor, fake_output: torch.Tensor
    ) -> torch.Tensor:
        """Discriminator wants to maximize real - fake."""
        return fake_output.mean() - real_output.mean()

    @staticmethod
    def gradient_penalty(
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        B = real.shape[0]
        alpha = torch.rand(B, 1, 1, 1, device=real.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_out = discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_out,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_out),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        return gp


class SteganoLoss(nn.Module):
    """
    Combined steganography training loss.

    Total = λ_img * ImageQuality + λ_msg * MessageLoss
          + λ_freq * FrequencyLoss + λ_adv * AdversarialLoss
    """

    def __init__(self, config=None):
        super().__init__()
        self.image_loss = ImageQualityLoss()
        self.message_loss = MessageLoss()
        self.frequency_loss = FrequencyLoss()

        if config:
            self.lambda_image = config.lambda_image
            self.lambda_message = config.lambda_message
            self.lambda_perceptual = config.lambda_perceptual
            self.lambda_adversarial = config.lambda_adversarial
            self.lambda_frequency = config.lambda_frequency
        else:
            self.lambda_image = 1.0
            self.lambda_message = 10.0
            self.lambda_perceptual = 0.5
            self.lambda_adversarial = 0.01
            self.lambda_frequency = 0.1

    def forward(
        self,
        stego: torch.Tensor,
        cover: torch.Tensor,
        decoded_msg: torch.Tensor,
        target_msg: torch.Tensor,
        disc_output: torch.Tensor = None,
    ) -> dict:
        losses = {}

        # Image quality losses
        img_losses = self.image_loss(stego, cover)
        losses.update(img_losses)

        # Message recovery loss
        losses["message"] = self.message_loss(decoded_msg, target_msg)

        # Frequency loss
        losses["frequency"] = self.frequency_loss(stego, cover)

        # Total encoder loss
        total = (
            self.lambda_image * img_losses["total_image"]
            + self.lambda_message * losses["message"]
            + self.lambda_frequency * losses["frequency"]
        )

        if disc_output is not None:
            adv_loss = WGANGPLoss.generator_loss(disc_output)
            losses["adversarial"] = adv_loss
            total += self.lambda_adversarial * adv_loss

        losses["total"] = total
        return losses
