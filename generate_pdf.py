#!/usr/bin/env python3
"""
Generate comprehensive PDF documentation for the Steganography Project.
Uses reportlab for professional PDF generation.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from datetime import datetime

def create_pdf():
    filename = "Steganography_Project_Complete_Flow.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)

    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#2e5fa3'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=14
    )

    # ========== TITLE PAGE ==========
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("AI-Enhanced Multi-Modal Secure Steganography System", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Complete Project Flow Documentation", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')}", body_style))
    story.append(Paragraph("<b>Institution:</b> Delhi Technological University", body_style))
    story.append(Paragraph("<b>Department:</b> Computer Science Engineering", body_style))
    story.append(Paragraph("<b>Project Type:</b> B.Tech Major Project (Project-II), AY 2025-26", body_style))
    story.append(PageBreak())

    # ========== SECTION 1: HIGH-LEVEL OVERVIEW ==========
    story.append(Paragraph("1. HIGH-LEVEL OVERVIEW", heading1_style))
    story.append(Paragraph(
        "This is a B.Tech major project that implements <b>multi-modal steganography</b> (hiding encrypted data in images, audio, and video) using both classical algorithms and deep learning. The primary focus is <b>video steganography</b> with 6 encoding methods across 3 modalities.",
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))

    # ========== SECTION 2: SYSTEM ARCHITECTURE ==========
    story.append(Paragraph("2. SYSTEM ARCHITECTURE", heading1_style))
    arch_text = """
    <font face="Courier" size="9">
    User ↓<br/>
    React Frontend (Vite + Tailwind)<br/>
    ↓<br/>
    FastAPI Backend (/api/image, /api/audio, /api/video endpoints)<br/>
    ↓<br/>
    Encryption Layer (AES-256-GCM)<br/>
    ↓<br/>
    Core Steganography Algorithms<br/>
    ↓<br/>
    Output Media + Quality Metrics
    </font>
    """
    story.append(Paragraph(arch_text, body_style))
    story.append(Spacer(1, 0.15*inch))

    # ========== SECTION 3: DATA FLOW ==========
    story.append(Paragraph("3. COMPLETE DATA FLOW", heading1_style))

    story.append(Paragraph("<b>ENCODING (HIDING) PROCESS:</b>", heading2_style))
    story.append(Paragraph(
        "<b>1. User Input</b> → Message text + Cover media + Password + Method choice<br/>"
        "<b>2. Encryption</b> → Generate random salt (16 bytes) → Derive key from password using PBKDF2-HMAC-SHA256 (600k iterations) → Encrypt message with AES-256-GCM (confidentiality + integrity) → Output: [salt(16B) | nonce(12B) | auth_tag(16B) | ciphertext]<br/>"
        "<b>3. Steganography</b> → Embed encrypted bytes into media using selected method:<br/>"
        "• LSB: Hide in least significant bits of pixel/sample values<br/>"
        "• DCT: Hide in discrete cosine transform coefficients<br/>"
        "• DWT: Hide in wavelet transform detail coefficients<br/>"
        "• Deep Learning (U-Net, HiDDeN, INN): Neural network encodes data end-to-end<br/>"
        "<b>4. For Video</b> → Extract frames → Use optical flow to detect motion → Embed only in stable regions → Temporal consistency checking<br/>"
        "<b>5. Output</b> → Stego media file + Quality metrics (PSNR, SSIM, LPIPS, etc.)",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>DECODING (REVEALING) PROCESS:</b>", heading2_style))
    story.append(Paragraph(
        "<b>1. User Input</b> → Stego media + Password + Same method used<br/>"
        "<b>2. Extract Encrypted Data</b> → Reverse steganography process to retrieve bytes<br/>"
        "<b>3. Decryption</b> → Parse salt, nonce, auth_tag → Derive same key from password + salt → Decrypt with AES-256-GCM (validates authentication tag = integrity check)<br/>"
        "<b>4. Output</b> → Original message (or error if tampered)",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 4: AVAILABLE METHODS ==========
    story.append(Paragraph("4. AVAILABLE METHODS & IMPLEMENTATIONS", heading1_style))

    story.append(Paragraph("<b>IMAGE STEGANOGRAPHY (/core/image/)</b>", heading2_style))
    image_table_data = [
        ['Method', 'File', 'Technique', 'Key Feature'],
        ['LSB', 'lsb.py', 'Embed in pixel LSBs', 'Randomized pixel order via seed'],
        ['DCT', 'dct_stego.py', 'Quantization Index Modulation on 8×8 blocks', 'Adaptive strength (α=10)'],
        ['DWT', 'dwt_stego.py', 'Multi-level Haar/Daubechies wavelet', 'Embed in detail coefficients'],
    ]
    image_table = Table(image_table_data, colWidths=[0.8*inch, 0.8*inch, 2.0*inch, 1.9*inch])
    image_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(image_table)
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>AUDIO STEGANOGRAPHY (/core/audio/)</b>", heading2_style))
    audio_table_data = [
        ['Method', 'File', 'Technique', 'Key Feature'],
        ['LSB', 'lsb.py', 'Embed in PCM sample LSBs', '16-bit audio samples'],
        ['DWT', 'dwt_stego.py', 'Daubechies-4 wavelet (4 levels)', 'Adaptive strength (α=0.02)'],
    ]
    audio_table = Table(audio_table_data, colWidths=[0.8*inch, 0.8*inch, 2.0*inch, 1.9*inch])
    audio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(audio_table)
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>VIDEO STEGANOGRAPHY (/core/video/) — PRIMARY FOCUS</b>", heading2_style))
    video_table_data = [
        ['Method', 'File', 'Technique', 'Key Feature'],
        ['LSB', 'lsb.py', 'Frame-by-frame LSB embedding', 'Motion compensation optional'],
        ['DCT', 'dct_stego.py', 'DCT on each frame + temporal awareness', '8×8 block-based'],
        ['DWT', 'dwt_stego.py', 'Wavelet per frame + frame selection', 'Skips every Nth frame'],
    ]
    video_table = Table(video_table_data, colWidths=[0.8*inch, 0.8*inch, 2.0*inch, 1.9*inch])
    video_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(video_table)
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "<b>Video-Specific Innovations:</b><br/>"
        "• <b>Optical Flow</b> (Farneback): Identifies low-motion regions (safe to embed)<br/>"
        "• <b>Frame Selection</b>: Skip frames (embed_every_n=2) to maintain temporal coherence<br/>"
        "• <b>Temporal Consistency</b>: Ensures no visible flicker across frames<br/>"
        "• <b>Codec Robustness</b>: Handles H.264/H.265 compression during encoding/decoding",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 5: FILES STRUCTURE ==========
    story.append(Paragraph("5. FILES STRUCTURE & ROLES", heading1_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"8\">steganography-project/<br/>"
        "├── config/<br/>"
        "│   └── settings.py               # Global config (encryption params, paths)<br/>"
        "├── core/                         # Core algorithms (classical methods)<br/>"
        "│   ├── encryption/<br/>"
        "│   │   ├── aes_cipher.py        # AES-256-GCM + PBKDF2<br/>"
        "│   │   └── integrity.py         # SHA-256 hashing<br/>"
        "│   ├── image/<br/>"
        "│   │   ├── lsb.py               # Image LSB<br/>"
        "│   │   ├── dct_stego.py         # Image DCT<br/>"
        "│   │   └── dwt_stego.py         # Image DWT<br/>"
        "│   ├── audio/<br/>"
        "│   │   ├── lsb.py               # Audio LSB<br/>"
        "│   │   └── dwt_stego.py         # Audio DWT<br/>"
        "│   ├── video/<br/>"
        "│   │   ├── frame_utils.py       # PyAV frame extraction, optical flow<br/>"
        "│   │   ├── lsb.py               # Video LSB + motion comp.<br/>"
        "│   │   ├── dct_stego.py         # Video DCT + temporal<br/>"
        "│   │   └── dwt_stego.py         # Video DWT + frame selection<br/>"
        "│   └── metrics/<br/>"
        "│       └── evaluate.py          # PSNR, SSIM, MS-SSIM, LPIPS, BER<br/>"
        "├── models/                       # Deep Learning Models (PyTorch)<br/>"
        "│   ├── layers.py                # CBAM, ConvNeXt, noise layers<br/>"
        "│   ├── losses.py                # Loss functions<br/>"
        "│   ├── train.py                 # Training pipeline<br/>"
        "│   ├── unet/<br/>"
        "│   │   └── encoder_decoder.py   # Attention U-Net++<br/>"
        "│   ├── hidden/<br/>"
        "│   │   └── hidden_model.py      # HiDDeN + WGAN-GP<br/>"
        "│   └── invertible/<br/>"
        "│       └── inn_model.py         # INN with Haar wavelet<br/>"
        "├── api/<br/>"
        "│   └── main.py                  # FastAPI backend<br/>"
        "├── frontend/                     # React + Vite + Tailwind<br/>"
        "├── scripts/                      # CLI utilities<br/>"
        "├── data/                        # Test media files<br/>"
        "└── PLANNING.md                  # Project planning</font>",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 6: DATASET ==========
    story.append(Paragraph("6. DATASET STRUCTURE", heading1_style))
    story.append(Paragraph(
        "<b>Where Data Lives:</b> /data/ directory<br/>"
        "• <b>Images:</b> data/images/ — PNG, JPG, BMP test images<br/>"
        "• <b>Audio:</b> data/audio/ — WAV files (16-bit PCM, 44.1 kHz)<br/>"
        "• <b>Videos:</b> data/videos/ — MP4, AVI, MKV files<br/><br/>"
        "<b>How Datasets Are Loaded:</b><br/>"
        "<font face=\"Courier\" size=\"8\">from models.train import StegoImageDataset, StegoVideoDataset<br/>"
        "<br/>"
        "# Image Dataset<br/>"
        "image_dataset = StegoImageDataset(<br/>"
        "    \"data/images\",<br/>"
        "    image_size=256,<br/>"
        "    transform=torchvision.transforms.Compose([...])<br/>"
        ")<br/>"
        "<br/>"
        "# Video Dataset<br/>"
        "video_dataset = StegoVideoDataset(<br/>"
        "    \"data/videos\",<br/>"
        "    frame_size=256,<br/>"
        "    temporal_window=5<br/>"
        ")<br/>"
        "<br/>"
        "loader = DataLoader(image_dataset, batch_size=8, shuffle=True)</font>",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 7: TRAINING DL MODELS ==========
    story.append(Paragraph("7. TRAINING DEEP LEARNING MODELS", heading1_style))

    story.append(Paragraph("<b>a) Attention U-Net++ (Image Focus)</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"8\">from models.unet import AttentionUNet++<br/>"
        "<br/>"
        "model = AttentionUNet++(<br/>"
        "    msg_length=128,<br/>"
        "    in_channels=3,<br/>"
        "    out_channels=3<br/>"
        ")</font><br/><br/>"
        "• <b>Encoder-Decoder</b> with dense skip connections<br/>"
        "• <b>CBAM attention</b> for spatial feature selection<br/>"
        "• <b>ConvNeXt blocks</b> for modern architecture<br/>"
        "• <b>Multi-scale message injection</b> at different depths",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>b) HiDDeN Adversarial (Robustness)</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"8\">from models.hidden import HiDDeNSteganography<br/>"
        "<br/>"
        "encoder = HiDDeNSteganography(msg_length=128)<br/>"
        "decoder = HiDDeNSteganography(msg_length=128, decoder=True)<br/>"
        "discriminator = HiDDeNDiscriminator()</font><br/><br/>"
        "• <b>Encoder + Decoder + WGAN-GP Discriminator</b><br/>"
        "• <b>Spectral normalization</b> in discriminator<br/>"
        "• <b>Noise layer</b> simulates: JPEG, Gaussian noise, crop, blur, codec<br/>"
        "• <b>Frequency loss</b> to minimize spectral anomalies",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>c) Invertible Neural Network (INN)</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"8\">from models.invertible import InvertibleSteganography<br/>"
        "<br/>"
        "model = InvertibleSteganography(<br/>"
        "    msg_length=128,<br/>"
        "    use_temporal_attention=True<br/>"
        ")</font><br/><br/>"
        "• <b>Normalizing flow</b> with affine coupling layers<br/>"
        "• <b>Haar wavelet lifting</b> for invertibility guarantee<br/>"
        "• <b>3D temporal attention</b> for video (motion tracking)<br/>"
        "• <b>Mathematically reversible</b>: Perfect cover pixel recovery",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 8: TRAINING PIPELINE ==========
    story.append(Paragraph("8. TRAINING PIPELINE", heading1_style))

    story.append(Paragraph("<b>Run Training:</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"8\">from models.train import train_hidden<br/>"
        "from torch.utils.data import DataLoader<br/>"
        "<br/>"
        "dataset = StegoImageDataset(\"data/images\", image_size=256)<br/>"
        "loader = DataLoader(dataset, batch_size=8, shuffle=True)<br/>"
        "<br/>"
        "model = HiDDeNSteganography(msg_length=128)<br/>"
        "train_hidden(<br/>"
        "    model,<br/>"
        "    train_loader=loader,<br/>"
        "    val_loader=val_loader,<br/>"
        "    epochs=100,<br/>"
        "    device=\"cuda\"<br/>"
        ")</font>",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Training Features:</b>", heading2_style))
    story.append(Paragraph(
        "• <b>Mixed Precision (AMP):</b> FP16 + FP32 for speed<br/>"
        "• <b>Scheduler:</b> Cosine annealing with warm restarts<br/>"
        "• <b>WandB Integration:</b> Real-time experiment tracking<br/>"
        "• <b>Multi-GPU:</b> Data parallel training support<br/>"
        "• <b>Checkpointing:</b> Save/resume best models<br/><br/>"
        "<b>Loss Functions (Combined):</b><br/>"
        "Total Loss = λ₁·MSE + λ₂·MS-SSIM + λ₃·LPIPS + λ₄·BCE(msg) + λ₅·Frequency + λ₆·WGAN-GP",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph(
        "• <b>MSE:</b> Pixel-level fidelity<br/>"
        "• <b>MS-SSIM:</b> Multi-scale structural similarity<br/>"
        "• <b>LPIPS:</b> Deep perceptual loss<br/>"
        "• <b>BCE:</b> Message recovery accuracy<br/>"
        "• <b>Frequency:</b> Spectral flatness<br/>"
        "• <b>WGAN-GP:</b> Adversarial training",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 9: TESTING & EVALUATION ==========
    story.append(Paragraph("9. TESTING & EVALUATION", heading1_style))

    story.append(Paragraph("<b>Metrics Computed (/core/metrics/evaluate.py):</b>", heading2_style))
    metrics_table_data = [
        ['Metric', 'Target', 'Meaning'],
        ['PSNR', '> 35 dB', 'Peak Signal-to-Noise Ratio'],
        ['SSIM', '> 0.97', 'Structural Similarity (1.0 = identical)'],
        ['MS-SSIM', '> 0.95', 'Multi-scale SSIM (perceptual)'],
        ['MSE', '< 10', 'Mean Squared Error'],
        ['SNR', '> 30 dB', 'Signal-to-Noise Ratio'],
        ['LPIPS', '< 0.05', 'Learned Perceptual Image Patch'],
        ['BER', '0%', 'Bit Error Rate (message corruption)'],
    ]
    metrics_table = Table(metrics_table_data, colWidths=[1.2*inch, 1.3*inch, 3.2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("<b>Test Flow:</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"8\">from core.metrics import compute_all_metrics<br/>"
        "<br/>"
        "stego_image = encoder.encode(cover_image, secret_data)<br/>"
        "metrics = compute_all_metrics(cover_image, stego_image)<br/>"
        "print(f\"PSNR: {metrics['psnr']:.2f} dB\")<br/>"
        "print(f\"SSIM: {metrics['ssim']:.4f}\")</font>",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 10: API ENDPOINTS ==========
    story.append(Paragraph("10. API ENDPOINTS (FastAPI)", heading1_style))

    story.append(Paragraph("<b>Backend running at:</b> http://localhost:8000", heading2_style))

    story.append(Paragraph(
        "<b>Endpoints:</b><br/>"
        "POST /api/image/encode       # Hide message in image<br/>"
        "POST /api/image/decode       # Extract message from image<br/>"
        "POST /api/audio/encode       # Hide message in audio<br/>"
        "POST /api/audio/decode       # Extract message from audio<br/>"
        "POST /api/video/encode       # Hide message in video<br/>"
        "POST /api/video/decode       # Extract message from video<br/>"
        "GET  /api/methods            # List available methods<br/>"
        "GET  /api/metrics            # Get evaluation results",
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("<b>Example Encode Request:</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"8\">curl -X POST http://localhost:8000/api/image/encode \\<br/>"
        "  -F \"cover=@input.jpg\" \\<br/>"
        "  -F \"message=Secret\" \\<br/>"
        "  -F \"password=mypassword\" \\<br/>"
        "  -F \"method=dwt\"</font>",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 11: FRONTEND ==========
    story.append(Paragraph("11. FRONTEND (React)", heading1_style))

    story.append(Paragraph("<b>Running at:</b> http://localhost:5173", heading2_style))

    story.append(Paragraph(
        "<b>Pages:</b><br/>"
        "• <b>Image:</b> Encode/Decode images with LSB/DCT/DWT/U-Net methods<br/>"
        "• <b>Audio:</b> Encode/Decode audio with LSB/DWT<br/>"
        "• <b>Video:</b> Encode/Decode video with LSB/DCT/DWT/INN/HiDDeN<br/>"
        "• <b>Metrics Dashboard:</b> Visualize PSNR, SSIM, BER graphs",
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("<b>User Flow:</b>", heading2_style))
    story.append(Paragraph(
        "1. Select modality (Image/Audio/Video)<br/>"
        "2. Upload cover media<br/>"
        "3. Enter secret message<br/>"
        "4. Enter password<br/>"
        "5. Choose method<br/>"
        "6. Click \"Encode\" → Download stego media + metrics<br/>"
        "7. To decode: Upload stego media + enter password",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 12: END-TO-END EXAMPLE ==========
    story.append(Paragraph("12. COMPLETE END-TO-END EXAMPLE", heading1_style))

    story.append(Paragraph("<b>CLI Demo:</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"8\">from core.image import ImageDWT<br/>"
        "from core.encryption import AESCipher<br/>"
        "from core.metrics import compute_all_metrics<br/>"
        "import cv2<br/>"
        "<br/>"
        "# 1. Read cover image<br/>"
        "cover = cv2.imread(\"data/images/test.jpg\")<br/>"
        "<br/>"
        "# 2. Encrypt message<br/>"
        "cipher = AESCipher(\"mypassword\")<br/>"
        "encrypted = cipher.encrypt_message(\"Secret message\")<br/>"
        "<br/>"
        "# 3. Embed in image using DWT<br/>"
        "encoder = ImageDWT(wavelet=\"haar\", level=2, alpha=5.0)<br/>"
        "stego = encoder.encode(cover, encrypted)<br/>"
        "<br/>"
        "# 4. Compute metrics<br/>"
        "metrics = compute_all_metrics(cover, stego)<br/>"
        "print(f\"PSNR: {metrics['psnr']:.2f} dB\")<br/>"
        "<br/>"
        "# 5. Save stego<br/>"
        "cv2.imwrite(\"output.jpg\", stego)<br/>"
        "<br/>"
        "# 6. Extract & Decrypt<br/>"
        "extracted = encoder.decode(stego)<br/>"
        "decrypted_msg = cipher.decrypt_message(extracted)<br/>"
        "print(f\"Recovered: {decrypted_msg}\")</font>",
        body_style
    ))
    story.append(PageBreak())

    # ========== SECTION 13: SECURITY PROPERTIES ==========
    story.append(Paragraph("13. SECURITY PROPERTIES", heading1_style))

    security_data = [
        ['Property', 'Implementation'],
        ['Confidentiality', 'AES-256-GCM (256-bit keys, authenticated)'],
        ['Integrity', 'GCM authentication tag + SHA-256 hash'],
        ['Key Derivation', 'PBKDF2-HMAC-SHA256 (600k iterations)'],
        ['Imperceptibility', 'Adaptive embedding + perceptual DL losses'],
        ['Anti-Steganalysis', 'HiDDeN adversarial + frequency optimization'],
    ]
    security_table = Table(security_data, colWidths=[2.0*inch, 4.7*inch])
    security_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(security_table)
    story.append(PageBreak())

    # ========== SECTION 14: RUN COMMANDS ==========
    story.append(Paragraph("14. RUN COMMANDS SUMMARY", heading1_style))

    story.append(Paragraph("<b>Backend Setup:</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"9\">cd steganography-project<br/>"
        "pip install -r requirements.txt<br/>"
        "python scripts/download_test_data.py<br/>"
        "uvicorn api.main:app --reload --port 8000</font>",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Frontend Setup:</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"9\">cd frontend<br/>"
        "npm install<br/>"
        "npm run dev  # Access at http://localhost:5173</font>",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>CLI Demo:</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"9\">python scripts/run_demo.py</font>",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))

    story.append(Paragraph("<b>Train Deep Learning Models:</b>", heading2_style))
    story.append(Paragraph(
        "<font face=\"Courier\" size=\"9\">python -c \"from models.train import train_hidden; ...\"</font>",
        body_style
    ))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph(
        "<b>CONCLUSION:</b><br/>"
        "This is a sophisticated, production-ready steganography system! The classical methods (LSB/DCT/DWT) provide a baseline, while deep learning models push robustness to state-of-the-art levels with adversarial training and perceptual losses.",
        body_style
    ))

    # Build PDF
    doc.build(story)
    print(f"✅ PDF generated successfully: {filename}")
    return filename

if __name__ == "__main__":
    create_pdf()
