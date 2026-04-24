"""
Stage 1: Four-Laser Source Characterization for CubeSat BB84 QKD
Wavelength stability, thermal drift, and spectral distinguishability analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations

# ─────────────────────────────────────────────
# LASER PARAMETERS
# ─────────────────────────────────────────────

# Each laser encodes one BB84 polarization state: H, V, D, AD
LASER_LABELS = ['H (0°)', 'V (90°)', 'D (45°)', 'AD (135°)']
LASER_COLORS = ['#4fc3f7', '#ef5350', '#66bb6a', '#ffa726']

# Nominal center wavelengths (nm) — ideally identical, but real diodes differ
# We'll model a small manufacturing spread around 1280 nm
LAMBDA_0_BASE = 1280.0  # nm
MANUFACTURING_OFFSETS = np.array([0.0, 0.3, -0.2, 0.5])  # nm, realistic spread
LAMBDA_0 = LAMBDA_0_BASE + MANUFACTURING_OFFSETS

# Linewidth (nm) — fixed for the parameter sweep
LINEWIDTH = 2.0  # nm, mid-range of 1–5 nm

# Thermal drift coefficients (nm/°C) — typical for InGaAs laser diodes near 1280 nm
D_LAMBDA_DT = np.array([0.28, 0.30, 0.27, 0.29])  # nm/°C, slight variation per laser

# Current tuning (nm/mA) — small secondary effect
D_LAMBDA_DI = np.array([0.010, 0.011, 0.009, 0.010])  # nm/mA
DELTA_I = 0.0  # mA — assume fixed current drive for now

# CubeSat thermal environment: orbital period temperature swing
T_NOMINAL = 20.0   # °C
T_MIN     = -20.0  # °C  (eclipse)
T_MAX     = +60.0  # °C  (sunlit)
N_TEMP    = 500

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def wavelength(lambda_0, dl_dT, dT, dl_dI=0, dI=0):
    """Center wavelength as a function of temperature and current deviation."""
    return lambda_0 + dl_dT * dT + dl_dI * dI

def gaussian_spectrum(lam_axis, center, linewidth, amplitude=1.0):
    """Normalized Gaussian spectral profile."""
    sigma = linewidth / (2 * np.sqrt(2 * np.log(2)))  # FWHM → sigma
    return amplitude * np.exp(-0.5 * ((lam_axis - center) / sigma) ** 2)

def spectral_overlap(center_a, center_b, linewidth):
    """
    Bhattacharyya coefficient between two Gaussian spectra.
    = 1 → identical (indistinguishable)
    = 0 → no overlap (perfectly distinguishable)
    For equal linewidths: BC = exp(-d²/8σ²) where d = |center_a - center_b|
    """
    sigma = linewidth / (2 * np.sqrt(2 * np.log(2)))
    d = abs(center_a - center_b)
    return np.exp(-(d ** 2) / (8 * sigma ** 2))

def distinguishability(overlap):
    """
    Distinguishability D = 1 - overlap.
    D > 0.1 (10%) is our threshold: Eve gains non-negligible info.
    """
    return 1.0 - overlap

# ─────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────

T_range = np.linspace(T_MIN, T_MAX, N_TEMP)
dT_range = T_range - T_NOMINAL

# Wavelength trajectories over temperature
lambda_vs_T = np.zeros((4, N_TEMP))
for i in range(4):
    lambda_vs_T[i] = wavelength(LAMBDA_0[i], D_LAMBDA_DT[i], dT_range, D_LAMBDA_DI[i], DELTA_I)

# Pairwise distinguishability over temperature
pairs = list(combinations(range(4), 2))
pair_labels = [f"{LASER_LABELS[a]} vs {LASER_LABELS[b]}" for a, b in pairs]
dist_vs_T = np.zeros((len(pairs), N_TEMP))
for k, (a, b) in enumerate(pairs):
    for j in range(N_TEMP):
        ov = spectral_overlap(lambda_vs_T[a, j], lambda_vs_T[b, j], LINEWIDTH)
        dist_vs_T[k, j] = distinguishability(ov)

# ─────────────────────────────────────────────
# PARAMETER SWEEP: vary spread between lasers
# find crossover where distinguishability > threshold
# ─────────────────────────────────────────────

DIST_THRESHOLD = 0.10  # 10% — Eve gains non-negligible info
spread_values = np.linspace(0.0, 6.0, 500)  # nm spread across all 4 lasers
max_dist_vs_spread = np.zeros(len(spread_values))

for s_idx, spread in enumerate(spread_values):
    # Distribute spread evenly: lasers at [-3/2, -1/2, +1/2, +3/2] * (spread/3)
    offsets = np.linspace(-spread / 2, spread / 2, 4)
    centers = LAMBDA_0_BASE + offsets
    # At nominal temperature, compute worst-case pairwise distinguishability
    worst = 0.0
    for a, b in pairs:
        ov = spectral_overlap(centers[a], centers[b], LINEWIDTH)
        d = distinguishability(ov)
        if d > worst:
            worst = d
    max_dist_vs_spread[s_idx] = worst

# Find crossover point
crossover_idx = np.argmax(max_dist_vs_spread >= DIST_THRESHOLD)
crossover_spread = spread_values[crossover_idx] if max_dist_vs_spread[crossover_idx] >= DIST_THRESHOLD else None

# ─────────────────────────────────────────────
# SPECTRAL PLOT at three temperatures
# ─────────────────────────────────────────────

lam_axis = np.linspace(1270, 1292, 2000)
T_snapshots = [T_MIN, T_NOMINAL, T_MAX]
T_snapshot_idx = [np.argmin(np.abs(T_range - t)) for t in T_snapshots]

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 14), facecolor='#0a0e1a')

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.93, bottom=0.07)

ACCENT = '#00e5ff'
WARN   = '#ff6b35'
GRID_C = '#1e2a3a'
TEXT_C = '#cdd6f4'

plt.rcParams['text.color']   = TEXT_C
plt.rcParams['axes.labelcolor'] = TEXT_C
plt.rcParams['xtick.color']  = TEXT_C
plt.rcParams['ytick.color']  = TEXT_C

fig.suptitle('Stage 1 — Four-Laser Source Characterization\nCubeSat BB84 QKD @ 1280 nm',
             fontsize=16, color=ACCENT, fontweight='bold', y=0.98)

# ── Plot 1: Wavelength vs Temperature ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for i in range(4):
    ax1.plot(T_range, lambda_vs_T[i], color=LASER_COLORS[i],
             linewidth=2, label=LASER_LABELS[i])
ax1.axvline(T_NOMINAL, color='white', linestyle='--', alpha=0.4, linewidth=1)
ax1.axhspan(LAMBDA_0_BASE - LINEWIDTH/2, LAMBDA_0_BASE + LINEWIDTH/2,
            alpha=0.08, color=ACCENT, label=f'±Δλ/2 window ({LINEWIDTH} nm)')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Center Wavelength (nm)')
ax1.set_title('Wavelength Drift vs Temperature', color=TEXT_C, pad=8)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.2)
ax1.set_facecolor('#0d1117')
ax1.grid(color=GRID_C, linewidth=0.5)
ax1.tick_params(colors=TEXT_C)

# ── Plot 2: Pairwise Distinguishability vs Temperature ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
pair_colors = ['#ab47bc', '#26c6da', '#d4e157', '#ff7043', '#42a5f5', '#ec407a']
for k, label in enumerate(pair_labels):
    ax2.plot(T_range, dist_vs_T[k], color=pair_colors[k],
             linewidth=1.8, label=label, alpha=0.9)
ax2.axhline(DIST_THRESHOLD, color=WARN, linestyle='--', linewidth=1.5,
            label=f'Threshold ({int(DIST_THRESHOLD*100)}%)')
ax2.fill_between(T_range, DIST_THRESHOLD, dist_vs_T.max(axis=0),
                 where=dist_vs_T.max(axis=0) >= DIST_THRESHOLD,
                 alpha=0.12, color=WARN)
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Distinguishability D')
ax2.set_title('Pairwise Spectral Distinguishability vs Temperature', color=TEXT_C, pad=8)
ax2.legend(fontsize=7, loc='upper left', framealpha=0.2)
ax2.set_facecolor('#0d1117')
ax2.grid(color=GRID_C, linewidth=0.5)
ax2.set_ylim(0, 1)
ax2.tick_params(colors=TEXT_C)

# ── Plot 3: Parameter Sweep — Spread vs Max Distinguishability ─────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(spread_values, max_dist_vs_spread, color=ACCENT, linewidth=2.5)
ax3.axhline(DIST_THRESHOLD, color=WARN, linestyle='--', linewidth=1.5,
            label=f'Security threshold ({int(DIST_THRESHOLD*100)}%)')
if crossover_spread is not None:
    ax3.axvline(crossover_spread, color=WARN, linestyle=':', linewidth=1.5,
                label=f'Crossover at {crossover_spread:.2f} nm')
    ax3.fill_betweenx([0, 1], crossover_spread, spread_values[-1],
                      alpha=0.10, color=WARN)
    ax3.fill_betweenx([0, 1], 0, crossover_spread,
                      alpha=0.08, color='#00e676')
    ax3.text(crossover_spread / 2, 0.55, 'SECURE\nREGION',
             color='#00e676', fontsize=9, ha='center', alpha=0.8, fontweight='bold')
    ax3.text((crossover_spread + spread_values[-1]) / 2, 0.55, 'INSECURE\nREGION',
             color=WARN, fontsize=9, ha='center', alpha=0.8, fontweight='bold')
ax3.set_xlabel('Wavelength Spread Across 4 Lasers (nm)')
ax3.set_ylabel('Max Pairwise Distinguishability D')
ax3.set_title(f'Parameter Sweep: Spread vs Distinguishability\n(Linewidth fixed at {LINEWIDTH} nm)',
              color=TEXT_C, pad=8)
ax3.legend(fontsize=8, framealpha=0.2)
ax3.set_facecolor('#0d1117')
ax3.grid(color=GRID_C, linewidth=0.5)
ax3.set_ylim(0, 1)
ax3.set_xlim(0, spread_values[-1])
ax3.tick_params(colors=TEXT_C)

# ── Plot 4: Linewidth sweep — how threshold changes with Δλ ───────────────
ax4 = fig.add_subplot(gs[1, 1])
linewidth_sweep = np.linspace(0.5, 5.0, 200)
crossover_per_lw = []
for lw in linewidth_sweep:
    max_d_per_spread = []
    for spread in spread_values:
        offsets = np.linspace(-spread/2, spread/2, 4)
        centers = LAMBDA_0_BASE + offsets
        worst = max(distinguishability(spectral_overlap(centers[a], centers[b], lw))
                    for a, b in pairs)
        max_d_per_spread.append(worst)
    max_d_arr = np.array(max_d_per_spread)
    idx = np.argmax(max_d_arr >= DIST_THRESHOLD)
    crossover_per_lw.append(spread_values[idx] if max_d_arr[idx] >= DIST_THRESHOLD else spread_values[-1])

ax4.plot(linewidth_sweep, crossover_per_lw, color='#66bb6a', linewidth=2.5)
ax4.axvline(LINEWIDTH, color=ACCENT, linestyle='--', linewidth=1.5,
            label=f'Current Δλ = {LINEWIDTH} nm')
ax4.set_xlabel('Linewidth Δλ (nm)')
ax4.set_ylabel('Max Safe Spread (nm)')
ax4.set_title('Safe Wavelength Spread vs Linewidth\n(10% distinguishability threshold)',
              color=TEXT_C, pad=8)
ax4.legend(fontsize=8, framealpha=0.2)
ax4.set_facecolor('#0d1117')
ax4.grid(color=GRID_C, linewidth=0.5)
ax4.tick_params(colors=TEXT_C)

# ── Plot 5: Combined spectra at 3 temperatures ────────────────────────────
ax5 = fig.add_subplot(gs[2, :])
snapshot_styles = [('--', 0.5, 'Eclipse (−20°C)'),
                   ('-',  1.0, 'Nominal (+20°C)'),
                   (':',  0.7, 'Sunlit (+60°C)')]

for (ls, alpha, tlabel), tidx in zip(snapshot_styles, T_snapshot_idx):
    combined = np.zeros_like(lam_axis)
    for i in range(4):
        spec = gaussian_spectrum(lam_axis, lambda_vs_T[i, tidx], LINEWIDTH)
        combined += spec
        if ls == '-':  # only label individual lasers at nominal
            ax5.plot(lam_axis, spec, color=LASER_COLORS[i],
                     linewidth=1.2, alpha=0.5, linestyle=ls)
    combined /= combined.max()
    ax5.plot(lam_axis, combined, color='white' if ls == '-' else ACCENT,
             linewidth=2.5 if ls == '-' else 1.5,
             linestyle=ls, alpha=alpha,
             label=f'Combined — {tlabel}')

ax5.axhline(0.5, color=WARN, linestyle=':', linewidth=1, alpha=0.5,
            label='50% power level (peak resolution guide)')
ax5.set_xlabel('Wavelength (nm)')
ax5.set_ylabel('Normalized Spectral Power')
ax5.set_title('Combined Output Spectrum at Three Orbital Temperatures\n(Individual laser spectra shown at nominal T)',
              color=TEXT_C, pad=8)
ax5.legend(fontsize=8, framealpha=0.2, ncol=2)
ax5.set_facecolor('#0d1117')
ax5.grid(color=GRID_C, linewidth=0.5)
ax5.tick_params(colors=TEXT_C)

plt.savefig('/Users/neha/pulse-q/stage1_wavelength_stability.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────
print("=" * 60)
print("STAGE 1 SUMMARY — Wavelength Stability Analysis")
print("=" * 60)
print(f"Nominal wavelength:     {LAMBDA_0_BASE} nm")
print(f"Linewidth (fixed):      {LINEWIDTH} nm")
print(f"Thermal range:          {T_MIN}°C to {T_MAX}°C")
print()
print("Nominal center wavelengths (λ₀ + manufacturing offset):")
for i in range(4):
    print(f"  {LASER_LABELS[i]:10s}: {LAMBDA_0[i]:.2f} nm")
print()
print("Wavelength at temperature extremes:")
for i in range(4):
    lam_min = wavelength(LAMBDA_0[i], D_LAMBDA_DT[i], T_MIN - T_NOMINAL)
    lam_max = wavelength(LAMBDA_0[i], D_LAMBDA_DT[i], T_MAX - T_NOMINAL)
    print(f"  {LASER_LABELS[i]:10s}: {lam_min:.2f} nm (cold) → {lam_max:.2f} nm (hot)"
          f"  [drift: {lam_max - lam_min:.2f} nm]")
print()
print(f"Crossover spread (10% threshold, Δλ={LINEWIDTH}nm):", end=" ")
if crossover_spread:
    print(f"{crossover_spread:.2f} nm")
    print(f"  → Keep all 4 lasers within {crossover_spread:.2f} nm of each other")
else:
    print("Not reached within sweep range")
print()
print("Max pairwise distinguishability at nominal T:")
for k, (a, b) in enumerate(pairs):
    idx_nom = np.argmin(np.abs(T_range - T_NOMINAL))
    print(f"  {pair_labels[k]:30s}: {dist_vs_T[k, idx_nom]:.4f}")
print()
print("Max pairwise distinguishability at hot extreme (+60°C):")
idx_hot = np.argmin(np.abs(T_range - T_MAX))
for k, (a, b) in enumerate(pairs):
    flag = " ← EXCEEDS THRESHOLD" if dist_vs_T[k, idx_hot] >= DIST_THRESHOLD else ""
    print(f"  {pair_labels[k]:30s}: {dist_vs_T[k, idx_hot]:.4f}{flag}")
print()
print("Output saved to: stage1_wavelength_stability.png")