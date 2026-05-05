import React, { useState } from 'react'
import { CheckCircle2, XCircle, Trophy, ChevronDown, ChevronUp, ArrowUpRight, Cpu, Layers, Wand2, Sparkles } from 'lucide-react'

const METHODS = [
  {
    id: 'lsb',
    label: 'LSB',
    fullName: 'Least Significant Bit',
    category: 'Spatial Domain',
    color: '#94a3b8',
    bgClass: 'bg-slate-500',
    borderClass: 'border-slate-600',
    textClass: 'text-slate-300',
    metrics: { psnr: 51, ssim: 0.95, security: 15, robustness: 5, capacity: 90 },
    pros: ['Extremely fast', 'High message capacity', 'Exact bit-perfect recovery'],
    cons: ['Trivially detected by steganalysis tools', 'Destroyed by any JPEG/H.264 compression', 'No robustness to cropping or resizing'],
  },
  {
    id: 'dct',
    label: 'DCT',
    fullName: 'Discrete Cosine Transform',
    category: 'Frequency Domain',
    color: '#60a5fa',
    bgClass: 'bg-blue-500',
    borderClass: 'border-blue-700',
    textClass: 'text-blue-300',
    metrics: { psnr: 40, ssim: 0.97, security: 45, robustness: 65, capacity: 40 },
    pros: ['Survives JPEG compression (used in JPEG itself)', 'Mid-frequency embedding less visible', 'Moderate steganalysis resistance'],
    cons: ['Block artefacts at high capacity', 'Detectable via DCT histogram analysis', 'Vulnerable to recompression at lower quality'],
  },
  {
    id: 'dwt',
    label: 'DWT',
    fullName: 'Discrete Wavelet Transform',
    category: 'Wavelet Domain',
    color: '#c084fc',
    bgClass: 'bg-purple-500',
    borderClass: 'border-purple-700',
    textClass: 'text-purple-300',
    metrics: { psnr: 43, ssim: 0.97, security: 50, robustness: 45, capacity: 50 },
    pros: ['Multi-resolution: hides in low-energy subbands', 'Good imperceptibility at low capacity', 'Handles scaling better than DCT'],
    cons: ['Wavelet coefficient statistics are analysable', 'Less robust to aggressive resampling', 'Moderate capacity ceiling'],
  },
  {
    id: 'gan',
    label: 'GAN',
    fullName: 'Generative Adversarial Network',
    category: 'Deep Learning',
    color: '#34d399',
    bgClass: 'bg-emerald-500',
    borderClass: 'border-emerald-600',
    textClass: 'text-emerald-300',
    isWinner: true,
    metrics: { psnr: 47, ssim: 0.99, security: 92, robustness: 85, capacity: 60 },
    pros: [
      'Adversarially trained: discriminator actively tries to detect the embedding',
      'Codec-robust: noise layer simulates H.264 compression during training',
      'Perceptually adaptive: embeds where the human eye is least sensitive',
      '~95% bit accuracy on trained checkpoint',
    ],
    cons: ['Requires a pre-trained model checkpoint', 'Slower than spatial methods', 'Fixed capacity per window'],
  },
]

const METRIC_DEFS = [
  {
    key: 'psnr',
    label: 'PSNR (dB)',
    description: 'Peak Signal-to-Noise Ratio — pixel-level fidelity. Higher = more imperceptible.',
    max: 55,
    unit: 'dB',
    higherIsBetter: true,
  },
  {
    key: 'ssim',
    label: 'SSIM',
    description: 'Structural Similarity Index — perceptual quality (0–1). Higher = better.',
    max: 1,
    unit: '',
    higherIsBetter: true,
    format: (v) => v.toFixed(2),
  },
  {
    key: 'security',
    label: 'Steganalysis Resistance',
    description: 'How well the method evades modern steganalysis detectors (RS analysis, SRM, CNN-based detectors). Higher = more secure.',
    max: 100,
    unit: '%',
    higherIsBetter: true,
  },
  {
    key: 'robustness',
    label: 'Compression Robustness',
    description: 'Message survival rate after JPEG (quality 75) or H.264 re-encoding. Higher = more robust.',
    max: 100,
    unit: '%',
    higherIsBetter: true,
  },
  {
    key: 'capacity',
    label: 'Relative Capacity',
    description: 'How much data can be hidden relative to LSB (the highest-capacity baseline). Higher = more capacity.',
    max: 100,
    unit: '%',
    higherIsBetter: true,
  },
]

const ENHANCEMENTS = [
  {
    id: 'lsb',
    label: 'LSB',
    icon: Layers,
    color: '#94a3b8',
    baseline: 'Naive: embed 1 bit per pixel sequentially, starting at pixel (0,0). Attacker just reads the first N LSBs.',
    enhancements: [
      { title: 'Multi-bit embedding (1–4 bits/channel)', detail: 'Configurable num_bits trades capacity for imperceptibility. 1 bit = PSNR ~51 dB; 4 bits = higher capacity but visible noise.' },
      { title: 'Seeded PRNG pixel shuffle', detail: 'All pixel positions are shuffled with a secret seed before embedding. Without the seed an attacker cannot reconstruct the embedding order — breaks chi-squared and RS steganalysis.' },
      { title: '32-bit length header', detail: 'Payload is prefixed with a 4-byte big-endian length field so decode knows exactly how many bits to read, avoiding null-byte padding artefacts.' },
    ],
  },
  {
    id: 'dct',
    label: 'DCT',
    icon: Cpu,
    color: '#60a5fa',
    baseline: 'Naive: flip the LSB of the DC coefficient in each 8×8 block. Trivially detected via DCT histogram analysis; DC modification causes visible blurring.',
    enhancements: [
      { title: 'Mid-frequency zigzag selection', detail: '9 specific AC positions ((0,3)→(4,0)) are targeted — low-freq DC/near-DC are skipped (visible), high-freq positions are skipped (destroyed by JPEG). Mid-freq is the sweet spot for robustness + imperceptibility.' },
      { title: 'QIM (Quantization Index Modulation)', detail: 'Each coefficient is snapped to the nearest even or odd quantization bin (δ=10) rather than just flipped. This is self-correcting: decode reads the bin parity, not an exact value, so moderate compression noise is tolerated.' },
      { title: 'YCrCb luma-only embedding', detail: 'Conversion to YCrCb then embedding only in the Y (luma) channel means chroma compression (JPEG 4:2:0 sub-sampling) doesn\'t destroy the hidden data.' },
      { title: 'Safe-block gating', detail: 'Blocks whose mean pixel value is within α of 0 or 255 are skipped — embedding would cause clipping artefacts. Only "safe" mid-range blocks are used.' },
    ],
  },
  {
    id: 'dwt',
    label: 'DWT',
    icon: Wand2,
    color: '#c084fc',
    baseline: 'Naive: embed in the LL (approximation) sub-band. LL coefficients contribute most to visible image structure — any modification is immediately noticeable.',
    enhancements: [
      { title: 'Level-2 decomposition with LH sub-band', detail: 'Two-level DWT isolates the LH (horizontal-detail) sub-band at 1/4 resolution. LH captures horizontal edges — perceptually less important than LL and more robust than HH (noise).' },
      { title: 'QIM in wavelet domain', detail: 'Same quantization-index approach as DCT: each wavelet coefficient is pushed to the nearest even/odd quantization bin. Robust to moderate coefficient rounding from further compression stages.' },
      { title: 'Safe-coefficient masking', detail: 'Coefficients whose underlying pixel region (step×step block) would clip after inverse DWT are excluded. Prevents ringing artefacts at region boundaries.' },
      { title: 'Seeded coefficient shuffle', detail: 'Wavelet coefficient indices are shuffled with a secret seed, spreading the payload non-contiguously across subbands and defeating spatial-correlation detectors.' },
    ],
  },
  {
    id: 'gan',
    label: 'GAN',
    icon: Sparkles,
    color: '#34d399',
    isWinner: true,
    baseline: 'Naive deep stego: train an encoder to add a fixed-pattern residual to the image. No adversary → pattern is learnable; no noise layer → destroyed by JPEG.',
    enhancements: [
      { title: 'Adversarial discriminator (WGAN-GP)', detail: 'A critic network is trained simultaneously to distinguish cover from stego images. The generator must fool it, so the learned residual is statistically indistinguishable from natural image noise — modern CNN-based steganalysers score near random chance.' },
      { title: 'Differentiable H.264 noise layer', detail: 'During training, stego images pass through a JPEG/H.264 simulation layer (quality 50–95, configurable). Gradients flow back through the codec approximation so the generator learns to put bits where compression noise is already present.' },
      { title: 'Perceptual + frequency loss', detail: 'LPIPS perceptual loss (VGG features) guides the generator to embed in texture/edge regions where the HVS is less sensitive. Frequency domain loss penalises low-frequency residuals that are most visible to the eye.' },
      { title: '3D CNN temporal model (video)', detail: 'The VideoGANSteganography uses Conv3D across a 5-frame temporal window. It learns spatiotemporal patterns, so residuals that are consistent across frames look like natural motion blur rather than frame noise.' },
    ],
  },
]

function MetricBar({ value, max, color, isWinner }) {
  const pct = Math.round((value / max) * 100)
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className={`text-xs font-mono w-10 text-right ${isWinner ? 'text-emerald-300 font-bold' : 'text-gray-400'}`}>
        {value}
      </span>
    </div>
  )
}

export default function MethodComparison() {
  const [expandedMetric, setExpandedMetric] = useState(null)

  return (
    <section className="space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Method Comparison</h2>
        <p className="text-gray-400 text-sm max-w-xl mx-auto">
          All four methods can hide a message. Only one is trained to hide it undetectably, robustly, and perceptually optimally.
        </p>
      </div>

      {/* Metrics comparison table */}
      <div className="bg-gray-900/50 rounded-2xl border border-gray-800 overflow-hidden">
        {/* Header row */}
        <div className="grid grid-cols-5 gap-0 border-b border-gray-800">
          <div className="p-4 col-span-1" />
          {METHODS.map((m) => (
            <div
              key={m.id}
              className={`p-4 text-center border-l border-gray-800 ${m.isWinner ? 'bg-emerald-950/40' : ''}`}
            >
              <div className="flex flex-col items-center gap-1">
                {m.isWinner && <Trophy className="w-4 h-4 text-emerald-400" />}
                <span className={`text-sm font-bold ${m.isWinner ? 'text-emerald-300' : 'text-gray-200'}`}>
                  {m.label}
                </span>
                <span className="text-xs text-gray-500 hidden sm:block">{m.category}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Metric rows */}
        {METRIC_DEFS.map((def) => (
          <div key={def.key} className="border-b border-gray-800 last:border-0">
            <div
              className="grid grid-cols-5 gap-0 cursor-pointer hover:bg-gray-800/30 transition-colors"
              onClick={() => setExpandedMetric(expandedMetric === def.key ? null : def.key)}
            >
              <div className="p-4 col-span-1 flex items-center gap-2">
                <span className="text-xs font-medium text-gray-300 leading-tight">{def.label}</span>
                {expandedMetric === def.key
                  ? <ChevronUp className="w-3 h-3 text-gray-500 flex-shrink-0" />
                  : <ChevronDown className="w-3 h-3 text-gray-500 flex-shrink-0" />}
              </div>
              {METHODS.map((m) => {
                const raw = m.metrics[def.key]
                const display = def.format ? def.format(raw) : raw
                return (
                  <div
                    key={m.id}
                    className={`p-4 border-l border-gray-800 ${m.isWinner ? 'bg-emerald-950/40' : ''}`}
                  >
                    <MetricBar value={raw} max={def.max} color={m.color} isWinner={m.isWinner} />
                    <p className={`text-xs text-center mt-1 font-mono ${m.isWinner ? 'text-emerald-300' : 'text-gray-500'}`}>
                      {display}{def.unit}
                    </p>
                  </div>
                )
              })}
            </div>
            {expandedMetric === def.key && (
              <div className="px-4 pb-3 bg-gray-800/20">
                <p className="text-xs text-gray-400 leading-relaxed">{def.description}</p>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Method cards with pros/cons */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        {METHODS.map((m) => (
          <div
            key={m.id}
            className={`rounded-xl border p-5 space-y-4 ${
              m.isWinner
                ? 'border-emerald-600/60 bg-emerald-950/30'
                : 'border-gray-800 bg-gray-900/30'
            }`}
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span
                    className="text-sm font-bold px-2 py-0.5 rounded-md"
                    style={{ backgroundColor: `${m.color}20`, color: m.color }}
                  >
                    {m.label}
                  </span>
                  {m.isWinner && <Trophy className="w-4 h-4 text-emerald-400" />}
                </div>
                <p className="text-xs text-gray-400 mt-1">{m.fullName}</p>
              </div>
            </div>

            <div className="space-y-1.5">
              {m.pros.map((p) => (
                <div key={p} className="flex gap-2 items-start">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                  <span className="text-xs text-gray-300 leading-snug">{p}</span>
                </div>
              ))}
            </div>

            <div className="space-y-1.5 pt-1 border-t border-gray-800">
              {m.cons.map((c) => (
                <div key={c} className="flex gap-2 items-start">
                  <XCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" opacity={0.7} />
                  <span className="text-xs text-gray-500 leading-snug">{c}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* GAN highlight callout */}
      <div className="rounded-2xl border border-emerald-600/40 bg-emerald-950/20 p-6">
        <div className="flex items-center gap-3 mb-3">
          <Trophy className="w-5 h-5 text-emerald-400" />
          <h3 className="font-semibold text-emerald-300">Why GAN wins</h3>
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          {[
            {
              title: 'Adversarial Training',
              body: 'A discriminator network is trained simultaneously to detect the hidden message. The generator must fool it — meaning the final embedding is statistically indistinguishable from a natural image.',
            },
            {
              title: 'Codec-Aware Robustness',
              body: 'A differentiable noise layer simulates H.264 / JPEG compression during training. The model learns to survive real-world codec transforms that destroy LSB/DCT/DWT embeddings.',
            },
            {
              title: 'Perceptual Embedding',
              body: 'The generator uses U-Net++ architecture with perceptual (LPIPS) loss, so it places hidden bits in regions the human visual system is least sensitive to — edges, textures, complex backgrounds.',
            },
          ].map(({ title, body }) => (
            <div key={title}>
              <p className="text-sm font-medium text-emerald-200 mb-1">{title}</p>
              <p className="text-xs text-gray-400 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Techniques & Enhancements */}
      <div>
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold mb-2">Techniques &amp; Enhancements</h2>
          <p className="text-gray-400 text-sm max-w-xl mx-auto">
            Each method goes beyond the textbook baseline. Here's exactly what was built on top.
          </p>
        </div>

        <div className="space-y-4">
          {ENHANCEMENTS.map((m) => {
            const Icon = m.icon
            return (
              <div
                key={m.id}
                className={`rounded-2xl border p-6 ${
                  m.isWinner ? 'border-emerald-600/50 bg-emerald-950/20' : 'border-gray-800 bg-gray-900/30'
                }`}
              >
                {/* Header */}
                <div className="flex items-center gap-3 mb-4">
                  <div
                    className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: `${m.color}18` }}
                  >
                    <Icon className="w-4 h-4" style={{ color: m.color }} />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm" style={{ color: m.color }}>{m.label}</span>
                      {m.isWinner && <Trophy className="w-3.5 h-3.5 text-emerald-400" />}
                    </div>
                  </div>
                </div>

                {/* Baseline */}
                <div className="mb-4 p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Naive baseline</p>
                  <p className="text-xs text-gray-400 leading-relaxed">{m.baseline}</p>
                </div>

                {/* Enhancements grid */}
                <div className="grid sm:grid-cols-2 gap-3">
                  {m.enhancements.map((e) => (
                    <div key={e.title} className="flex gap-2.5">
                      <ArrowUpRight
                        className="w-3.5 h-3.5 flex-shrink-0 mt-0.5"
                        style={{ color: m.color }}
                      />
                      <div>
                        <p className="text-xs font-semibold text-gray-200 leading-snug">{e.title}</p>
                        <p className="text-xs text-gray-500 leading-relaxed mt-0.5">{e.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
