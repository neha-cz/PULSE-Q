"""
Stage 3: Combined Spectrum & Temporal Distinguishability
Four-laser BB84 CubeSat QKD system @ 1280 nm

Checks:
  1. Spectral shape: do the 4 Gaussian peaks merge into one unresolvable distribution?
  2. Temporal shape: are pulse envelopes matched across all 4 lasers?
  3. Joint spectro-temporal distinguishability: Eve's information gain combining both channels
  4. Resolution threshold: what spectrometer resolving power would Eve need to distinguish lasers?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations


# ─────────────────────────────────────────────
# LASER PARAMETERS (from Stages 1 & 2)
# ─────────────────────────────────────────────

LASER_LABELS = ['H  (0°)', 'V  (90°)', 'D  (45°)', 'AD (135°)']
LASER_COLORS = ['#4fc3f7', '#ef5350', '#66bb6a', '#ffa726']

LAMBDA_0_BASE       = 1280.0                          # nm
MANUFACTURING_OFFSETS = np.array([0.0, 0.3, -0.2, 0.5])  # nm (from Stage 1)
LAMBDA_0            = LAMBDA_0_BASE + MANUFACTURING_OFFSETS

LINEWIDTH_NM        = 2.0    # nm FWHM (fixed, from Stage 1)
DIST_THRESHOLD      = 0.10   # 10% distinguishability security threshold

# Thermal parameters
T_NOMINAL = 20.0
T_MIN, T_MAX = -20.0, 60.0
D_LAMBDA_DT = np.array([0.28, 0.30, 0.27, 0.29])   # nm/°C

# ─────────────────────────────────────────────
# TEMPORAL PULSE PARAMETERS
# Each laser diode has slightly different pulse characteristics
# due to manufacturing variation in driver circuits and diode capacitance
# ─────────────────────────────────────────────

# Pulse width (FWHM, nanoseconds) — nominally matched, but vary slightly
PULSE_WIDTH_NS_NOM  = 10.0   # ns nominal (adjustable, e.g. NPL98B range)
PULSE_WIDTH_OFFSETS = np.array([0.0, 0.4, -0.3, 0.6])   # ns variation per laser

# Pulse arrival time jitter/offset (ns) — timing misalignment at combiner
PULSE_TIMING_OFFSETS = np.array([0.0, 0.15, -0.10, 0.25])  # ns

# Pulse amplitude (normalized) — slight power variation per laser
PULSE_AMPLITUDE = np.array([1.00, 0.96, 1.03, 0.98])

# Rise time (ns) — affects pulse shape asymmetry
RISE_TIME_NS = np.array([0.8, 1.1, 0.9, 1.0])

# ─────────────────────────────────────────────
# SPECTRAL RESOLUTION PARAMETERS
# Eve's hypothetical measurement instruments
# ─────────────────────────────────────────────

# Resolving power R = λ/δλ — what Eve's spectrometer can resolve
EVE_RESOLVING_POWERS = [100, 500, 1000, 5000, 10000]

# Timing resolution of Eve's detector (ns)
EVE_TIMING_RESOLUTION_NS = [0.05, 0.1, 0.5, 1.0, 5.0]

# ─────────────────────────────────────────────
# WAVELENGTH AND TIME AXES
# ─────────────────────────────────────────────

lam_axis  = np.linspace(1268, 1295, 5000)   # nm
time_axis = np.linspace(-40, 40, 5000)       # ns, centered on nominal pulse

# Temperature snapshots
T_snapshots    = [T_MIN, T_NOMINAL, T_MAX]
T_snap_labels  = ['Eclipse (−20°C)', 'Nominal (+20°C)', 'Sunlit (+60°C)']
T_snap_colors  = ['#89b4fa', '#cdd6f4', '#fab387']

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def gaussian_spectrum(lam, center, fwhm, amplitude=1.0):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-0.5 * ((lam - center) / sigma)**2)

def center_wavelength_at_T(laser_idx, T):
    return LAMBDA_0[laser_idx] + D_LAMBDA_DT[laser_idx] * (T - T_NOMINAL)

def gaussian_pulse(t, center_ns, fwhm_ns, amplitude=1.0, rise_time_ns=1.0):
    """
    Asymmetric pulse: Gaussian with slight leading-edge shaping
    to model real laser diode turn-on behavior.
    """
    sigma = fwhm_ns / (2 * np.sqrt(2 * np.log(2)))
    base  = amplitude * np.exp(-0.5 * ((t - center_ns) / sigma)**2)
    # Asymmetry: suppress leading edge slightly
    asymmetry = 1.0 - 0.15 * np.exp(-(t - center_ns + 2*sigma)**2 / (2*rise_time_ns**2))
    asymmetry = np.clip(asymmetry, 0, 1)
    return base * asymmetry

def bhattacharyya_coefficient(spec_a, spec_b):
    """
    Bhattacharyya coefficient between two spectral distributions.
    = 1 → identical, 0 → no overlap.
    Normalized to treat spectra as probability distributions.
    """
    norm_a = spec_a / (spec_a.sum() + 1e-30)
    norm_b = spec_b / (spec_b.sum() + 1e-30)
    return np.sum(np.sqrt(norm_a * norm_b))

def temporal_overlap(pulse_a, pulse_b):
    """Cosine similarity between pulse shapes. Range [0,1]."""
    denom = np.linalg.norm(pulse_a) * np.linalg.norm(pulse_b)
    if denom < 1e-30:
        return 0.0
    return np.dot(pulse_a, pulse_b) / denom

def eve_spectral_resolution_nm(resolving_power, center_nm=LAMBDA_0_BASE):
    """Convert resolving power R = λ/δλ to wavelength resolution in nm."""
    return center_nm / resolving_power

def spectral_distinguishability_after_eve_filter(centers, fwhm, eve_res_nm):
    """
    After Eve applies a spectrometer with resolution eve_res_nm,
    does the combined spectrum show resolvable peaks?
    Model: convolve each Gaussian with Eve's instrument function (also Gaussian).
    Effective linewidth = sqrt(laser_fwhm² + eve_res²)
    """
    effective_fwhm = np.sqrt(fwhm**2 + eve_res_nm**2)
    specs = [gaussian_spectrum(lam_axis, c, effective_fwhm) for c in centers]
    # Check whether any two peaks are resolvable (Rayleigh criterion: separation > effective_fwhm)
    max_sep = max(abs(centers[a] - centers[b]) for a, b in combinations(range(4), 2))
    return max_sep, effective_fwhm, max_sep < effective_fwhm   # True = unresolvable (secure)

# ─────────────────────────────────────────────
# BUILD SPECTRA AT EACH TEMPERATURE
# ─────────────────────────────────────────────

spectra_by_T = {}   # {T: (individual_spectra [4 x N], combined)}
centers_by_T = {}

for T in T_snapshots:
    centers = np.array([center_wavelength_at_T(i, T) for i in range(4)])
    centers_by_T[T] = centers
    specs   = np.array([gaussian_spectrum(lam_axis, centers[i], LINEWIDTH_NM,
                                           PULSE_AMPLITUDE[i])
                         for i in range(4)])
    combined = specs.sum(axis=0)
    combined /= combined.max()
    spectra_by_T[T] = (specs, combined)

# ─────────────────────────────────────────────
# BUILD PULSE SHAPES FOR ALL 4 LASERS
# ─────────────────────────────────────────────

pulses = np.array([
    gaussian_pulse(time_axis,
                   center_ns   = PULSE_TIMING_OFFSETS[i],
                   fwhm_ns     = PULSE_WIDTH_NS_NOM + PULSE_WIDTH_OFFSETS[i],
                   amplitude   = PULSE_AMPLITUDE[i],
                   rise_time_ns= RISE_TIME_NS[i])
    for i in range(4)
])
combined_pulse = pulses.sum(axis=0)
combined_pulse /= combined_pulse.max()

# ─────────────────────────────────────────────
# PAIRWISE DISTINGUISHABILITIES
# ─────────────────────────────────────────────

pairs = list(combinations(range(4), 2))
pair_labels = [f"{LASER_LABELS[a].strip()} | {LASER_LABELS[b].strip()}" for a, b in pairs]

# Spectral distinguishability at each temperature
spec_dist_by_T = {}
for T in T_snapshots:
    specs, _ = spectra_by_T[T]
    dists = []
    for a, b in pairs:
        bc  = bhattacharyya_coefficient(specs[a], specs[b])
        dists.append(1.0 - bc)
    spec_dist_by_T[T] = np.array(dists)

# Temporal distinguishability (pulse shape overlap)
temp_dist = []
for a, b in pairs:
    ov  = temporal_overlap(pulses[a], pulses[b])  # cosine similarity, range [0,1]
    temp_dist.append(1.0 - ov)  # distinguishability = 1 - cosine_similarity
temp_dist = np.array(temp_dist)

# Joint distinguishability: Eve combines spectral AND temporal info
# D_joint = 1 - (1-D_spec)*(1-D_temp) — independence assumption
joint_dist_nom = 1.0 - (1.0 - spec_dist_by_T[T_NOMINAL]) * (1.0 - temp_dist)
joint_dist_hot = 1.0 - (1.0 - spec_dist_by_T[T_MAX])     * (1.0 - temp_dist)

# ─────────────────────────────────────────────
# EVE'S SPECTROMETER RESOLUTION SWEEP
# ─────────────────────────────────────────────

resolving_powers = np.logspace(1.5, 5, 300)
resolutions_nm   = LAMBDA_0_BASE / resolving_powers

# At each resolution: is the worst-case peak separation resolvable?
centers_nom = centers_by_T[T_NOMINAL]
centers_hot = centers_by_T[T_MAX]
max_sep_nom = max(abs(centers_nom[a]-centers_nom[b]) for a,b in pairs)
max_sep_hot = max(abs(centers_hot[a]-centers_hot[b]) for a,b in pairs)

# Rayleigh criterion: peaks resolved if separation > effective FWHM
effective_fwhm_nom = np.sqrt(LINEWIDTH_NM**2 + resolutions_nm**2)
effective_fwhm_hot = np.sqrt(LINEWIDTH_NM**2 + resolutions_nm**2)

resolvable_nom = max_sep_nom > effective_fwhm_nom
resolvable_hot = max_sep_hot > effective_fwhm_hot

# Crossover resolving power (Eve needs at least this to distinguish)
crossover_nom_idx = np.argmax(resolvable_nom)
crossover_hot_idx = np.argmax(resolvable_hot)
crossover_R_nom = resolving_powers[crossover_nom_idx] if resolvable_nom.any() else None
crossover_R_hot = resolving_powers[crossover_hot_idx] if resolvable_hot.any() else None

# ─────────────────────────────────────────────
# TEMPERATURE SWEEP: spectral spread and peak distinguishability
# ─────────────────────────────────────────────

T_sweep = np.linspace(T_MIN, T_MAX, 400)
max_spec_dist_vs_T = np.zeros(len(T_sweep))
wavelength_spread_vs_T = np.zeros(len(T_sweep))

for j, T in enumerate(T_sweep):
    centers = np.array([center_wavelength_at_T(i, T) for i in range(4)])
    wavelength_spread_vs_T[j] = centers.max() - centers.min()
    specs = np.array([gaussian_spectrum(lam_axis, centers[i], LINEWIDTH_NM) for i in range(4)])
    dists = [1.0 - bhattacharyya_coefficient(specs[a], specs[b]) for a, b in pairs]
    max_spec_dist_vs_T[j] = max(dists)

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 20), facecolor='#0a0e1a')
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35,
                        left=0.07, right=0.97, top=0.95, bottom=0.05)

ACCENT = '#00e5ff'
WARN   = '#ff6b35'
GOOD   = '#00e676'
GRID_C = '#1e2a3a'
TEXT_C = '#cdd6f4'
plt.rcParams.update({'text.color':TEXT_C, 'axes.labelcolor':TEXT_C,
                     'xtick.color':TEXT_C, 'ytick.color':TEXT_C})

fig.suptitle('Stage 3 — Combined Spectrum & Temporal Distinguishability\n'
             'Four-Laser BB84 CubeSat QKD @ 1280 nm | Eve\'s View',
             fontsize=16, color=ACCENT, fontweight='bold', y=0.975)

# ── Plot 1: Combined spectra at 3 temperatures ────────────────────────────
ax1 = fig.add_subplot(gs[0, :])   # full width
for idx, (T, Tlabel, Tcol) in enumerate(zip(T_snapshots, T_snap_labels, T_snap_colors)):
    specs, combined = spectra_by_T[T]
    ls = '-' if T == T_NOMINAL else ('--' if T == T_MIN else ':')
    lw = 2.8 if T == T_NOMINAL else 1.8
    # Individual laser spectra (only at nominal, for clarity)
    if T == T_NOMINAL:
        for i in range(4):
            s = specs[i] / specs[i].max()
            ax1.fill_between(lam_axis, 0, s*0.5, alpha=0.12, color=LASER_COLORS[i])
            ax1.plot(lam_axis, s*0.5, color=LASER_COLORS[i], linewidth=1.0,
                     alpha=0.6, linestyle='--', label=f'{LASER_LABELS[i].strip()} (individual)')
    ax1.plot(lam_axis, combined, color=Tcol, linewidth=lw, linestyle=ls,
             label=f'Combined — {Tlabel}', alpha=0.95)

ax1.axhline(0.5, color='white', linestyle=':', alpha=0.2, linewidth=1)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Normalized Spectral Power')
ax1.set_title("Eve's View: Combined Output Spectrum at Three Orbital Temperatures\n"
              "(Individual laser contributions shown at nominal T)",
              color=TEXT_C, pad=8)
ax1.legend(fontsize=8, framealpha=0.2, ncol=3)
ax1.set_facecolor('#0d1117')
ax1.grid(color=GRID_C, linewidth=0.5)
ax1.set_xlim(1272, 1290)

# ── Plot 2: Pulse shapes — temporal distinguishability ────────────────────
ax2 = fig.add_subplot(gs[1, 0])
for i in range(4):
    ax2.plot(time_axis, pulses[i], color=LASER_COLORS[i],
             linewidth=2.0, label=f'{LASER_LABELS[i].strip()}  '
                                   f'(Δt={PULSE_TIMING_OFFSETS[i]:+.2f} ns, '
                                   f'W={PULSE_WIDTH_NS_NOM+PULSE_WIDTH_OFFSETS[i]:.1f} ns)')
ax2.plot(time_axis, combined_pulse, color='white', linewidth=2.5,
         linestyle='--', label='Combined (normalized)', zorder=5)
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Normalized Power')
ax2.set_title('Pulse Shapes — Temporal Distinguishability\n'
              '(mismatches in timing, width, and rise time)',
              color=TEXT_C, pad=8)
ax2.legend(fontsize=7.5, framealpha=0.2)
ax2.set_facecolor('#0d1117')
ax2.grid(color=GRID_C, linewidth=0.5)
ax2.set_xlim(-20, 20)

# ── Plot 3: Pairwise distinguishability — spectral + temporal + joint ──────
ax3 = fig.add_subplot(gs[1, 1])
x     = np.arange(len(pairs))
width = 0.25
pair_cols = ['#ab47bc','#26c6da','#d4e157','#ff7043','#42a5f5','#ec407a']

bars1 = ax3.bar(x - width, spec_dist_by_T[T_NOMINAL]*100,
                width, label='Spectral D (nominal T)', color=[c+'aa' for c in pair_cols])
bars2 = ax3.bar(x,          temp_dist*100,
                width, label='Temporal D', color=[c+'66' for c in pair_cols])
bars3 = ax3.bar(x + width,  joint_dist_nom*100,
                width, label='Joint D (spectral + temporal)', color=pair_cols)

ax3.axhline(DIST_THRESHOLD*100, color=WARN, linestyle='--', linewidth=1.5,
            label=f'Security threshold ({DIST_THRESHOLD*100:.0f}%)')
ax3.set_xticks(x)
ax3.set_xticklabels([pl.replace(' | ', '\n') for pl in pair_labels], fontsize=7)
ax3.set_ylabel('Distinguishability (%)')
ax3.set_title('Pairwise Distinguishability: Spectral vs Temporal vs Joint\n'
              '(nominal temperature)',
              color=TEXT_C, pad=8)
ax3.legend(fontsize=7.5, framealpha=0.2)
ax3.set_facecolor('#0d1117')
ax3.grid(color=GRID_C, linewidth=0.5, axis='y')

# ── Plot 4: Eve's spectrometer resolution sweep ────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.semilogx(resolving_powers, effective_fwhm_nom, color=ACCENT,
             linewidth=2.5, label="Effective FWHM (Eve's instrument + laser)")
ax4.axhline(max_sep_nom, color=GOOD, linestyle='--', linewidth=1.8,
            label=f'Max laser separation @ nominal T ({max_sep_nom:.2f} nm)')
ax4.axhline(max_sep_hot, color=WARN, linestyle='--', linewidth=1.8,
            label=f'Max laser separation @ hot ({max_sep_hot:.2f} nm)')
if crossover_R_nom:
    ax4.axvline(crossover_R_nom, color=GOOD, linestyle=':', linewidth=1.5,
                label=f'Eve needs R > {crossover_R_nom:.0f} (nominal T)')
if crossover_R_hot:
    ax4.axvline(crossover_R_hot, color=WARN, linestyle=':', linewidth=1.5,
                label=f'Eve needs R > {crossover_R_hot:.0f} (hot)')

# Shade: region where Eve cannot resolve peaks
ax4.fill_between(resolving_powers,
                 effective_fwhm_nom, max_sep_hot,
                 where=~resolvable_nom, alpha=0.12, color=GOOD,
                 label='Peaks unresolvable (secure)')
ax4.set_xlabel("Eve's Spectrometer Resolving Power R = λ/δλ")
ax4.set_ylabel('Wavelength (nm)')
ax4.set_title("Eve's Spectrometer Resolution Threshold\n"
              "(resolving power needed to distinguish individual lasers)",
              color=TEXT_C, pad=8)
ax4.legend(fontsize=7.5, framealpha=0.2)
ax4.set_facecolor('#0d1117')
ax4.grid(color=GRID_C, linewidth=0.5)

# ── Plot 5: Spectral spread and max distinguishability vs temperature ──────
ax5 = fig.add_subplot(gs[2, 1])
ax5_twin = ax5.twinx()

l1, = ax5.plot(T_sweep, wavelength_spread_vs_T, color=ACCENT,
               linewidth=2.5, label='Wavelength spread (nm)')
l2, = ax5_twin.plot(T_sweep, max_spec_dist_vs_T*100, color=WARN,
                    linewidth=2.5, linestyle='--', label='Max pairwise D (%)')

ax5.axvline(T_NOMINAL, color='white', linestyle='--', alpha=0.3, linewidth=1)
ax5_twin.axhline(DIST_THRESHOLD*100, color='#ff1744', linestyle=':', linewidth=1.5,
                 label=f'{DIST_THRESHOLD*100:.0f}% threshold')

ax5.set_xlabel('Temperature (°C)')
ax5.set_ylabel('Wavelength Spread (nm)', color=ACCENT)
ax5_twin.set_ylabel('Max Distinguishability (%)', color=WARN)
ax5.tick_params(axis='y', colors=ACCENT)
ax5_twin.tick_params(axis='y', colors=WARN)
ax5.set_title('Spectral Spread & Distinguishability vs Temperature\n'
              '(shows thermal regime where security degrades)',
              color=TEXT_C, pad=8)
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax5.legend(lines, labels, fontsize=8, framealpha=0.2)
ax5.set_facecolor('#0d1117')
ax5.grid(color=GRID_C, linewidth=0.5)

# ── Plot 6: Pulse timing mismatch — zoom on distinguishable region ─────────
ax6 = fig.add_subplot(gs[3, 0])
# Show difference between each pulse and the reference (H laser)
ref_pulse_norm = pulses[0] / pulses[0].max()
for i in range(1, 4):
    p_norm = pulses[i] / pulses[i].max()
    diff   = p_norm - ref_pulse_norm
    ax6.plot(time_axis, diff*100, color=LASER_COLORS[i],
             linewidth=1.8,
             label=f'{LASER_LABELS[i].strip()} − H  '
                   f'(Δt={PULSE_TIMING_OFFSETS[i]:+.2f} ns, '
                   f'ΔW={PULSE_WIDTH_OFFSETS[i]:+.1f} ns)')
ax6.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=1)
ax6.axhline(5, color=WARN, linestyle=':', alpha=0.6, linewidth=1, label='±5% mismatch')
ax6.axhline(-5, color=WARN, linestyle=':', alpha=0.6, linewidth=1)
ax6.set_xlabel('Time (ns)')
ax6.set_ylabel('Pulse Shape Difference (%)')
ax6.set_title('Pulse Shape Residuals (each laser vs H reference)\n'
              '(Eve can exploit these differences in time domain)',
              color=TEXT_C, pad=8)
ax6.legend(fontsize=8, framealpha=0.2)
ax6.set_facecolor('#0d1117')
ax6.grid(color=GRID_C, linewidth=0.5)
ax6.set_xlim(-15, 15)

# ── Plot 7: Stage 1+2+3 combined security summary ─────────────────────────
ax7 = fig.add_subplot(gs[3, 1])
ax7.set_facecolor('#0d1117')
ax7.axis('off')

summary_data = {
    'Stage 1 — Spectral Stability': [
        f'Max safe wavelength spread:  0.78 nm',
        f'Current spread (nominal T):  {max(MANUFACTURING_OFFSETS)-min(MANUFACTURING_OFFSETS):.2f} nm  ✓',
        f'Current spread (hot +60°C):  {max_sep_hot:.2f} nm  {"✓" if max_sep_hot < 0.78 else "✗ EXCEEDS"}',
        f'Thermal drift per laser:     ~22–24 nm over orbit',
    ],
    'Stage 2 — Polarization (PER)': [
        f'Critical misalignment:       5.72°',
        f'QBER floor (nominal T):      0.040%  ✓',
        f'QBER floor (hot +60°C):      0.299%  ✓',
        f'All lasers above 20 dB PER at nominal T  ✓',
    ],
    'Stage 3 — Combined Spectrum': [
        f'Peaks unresolvable at nom T: {"Yes ✓" if not resolvable_nom[crossover_nom_idx-1] else "No ✗"}',
        f'Peaks unresolvable even at R=100000 (nom T)  ✓',
        f'Max temporal mismatch:       {max(abs(PULSE_TIMING_OFFSETS)):.2f} ns timing offset',
        f'Max pulse width mismatch:    {max(abs(PULSE_WIDTH_OFFSETS)):.1f} ns width offset',
        f'Joint D (worst pair, nom T): {joint_dist_nom.max()*100:.2f}%  '
        f'{"✓" if joint_dist_nom.max() < DIST_THRESHOLD else "✗"}',
    ],
}

y_pos = 0.97
for stage, items in summary_data.items():
    ax7.text(0.02, y_pos, stage, fontsize=9.5, color=ACCENT,
             fontweight='bold', transform=ax7.transAxes)
    y_pos -= 0.07
    for item in items:
        col = GOOD if '✓' in item else (WARN if '✗' in item else TEXT_C)
        ax7.text(0.05, y_pos, item, fontsize=8.5, color=col,
                 transform=ax7.transAxes, fontfamily='monospace')
        y_pos -= 0.065
    y_pos -= 0.03

ax7.set_title('Stages 1–3 Security Summary', color=TEXT_C, pad=8)

plt.savefig('/mnt/user-data/outputs/stage3_combined_spectrum.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────
print("=" * 65)
print("STAGE 3 SUMMARY — Combined Spectrum & Temporal Distinguishability")
print("=" * 65)

print(f"\n── Spectral Check ──────────────────────────────────────────────")
print(f"Laser center wavelengths at nominal T ({T_NOMINAL}°C):")
for i in range(4):
    print(f"  {LASER_LABELS[i]:12s}: {centers_by_T[T_NOMINAL][i]:.2f} nm")
print(f"Total spread at nominal T: {max_sep_nom:.3f} nm")
print(f"Total spread at hot +60°C: {max_sep_hot:.3f} nm")
print(f"Linewidth (FWHM):          {LINEWIDTH_NM:.1f} nm")
resolvable_str_nom = ('NO — Eve needs R > '+str(int(crossover_R_nom))) if resolvable_nom.any() and crossover_R_nom else 'YES — unresolvable across all R ✓'
resolvable_str_hot = ('NO — Eve needs R > '+str(int(crossover_R_hot))) if resolvable_hot.any() and crossover_R_hot else 'YES — unresolvable across all R ✓'
print(f"Peaks unresolvable (Rayleigh, nominal T)? {resolvable_str_nom}")
print(f"Peaks unresolvable (Rayleigh, hot +60C)?  {resolvable_str_hot}")

print(f"\n── Spectral Pairwise Distinguishability (nominal T) ─────────────")
for k, (a, b) in enumerate(pairs):
    flag = " ← EXCEEDS THRESHOLD" if spec_dist_by_T[T_NOMINAL][k] >= DIST_THRESHOLD else " ✓"
    print(f"  {pair_labels[k]:30s}: {spec_dist_by_T[T_NOMINAL][k]*100:.3f}%{flag}")

print(f"\n── Temporal Check ───────────────────────────────────────────────")
print(f"Pulse width (nominal): {PULSE_WIDTH_NS_NOM} ns")
print(f"Pulse parameters per laser:")
for i in range(4):
    print(f"  {LASER_LABELS[i]:12s}: width={PULSE_WIDTH_NS_NOM+PULSE_WIDTH_OFFSETS[i]:.1f} ns  "
          f"timing_offset={PULSE_TIMING_OFFSETS[i]:+.2f} ns  "
          f"amplitude={PULSE_AMPLITUDE[i]:.2f}")

print(f"\nTemporal pairwise distinguishability:")
for k, (a, b) in enumerate(pairs):
    flag = " ← EXCEEDS THRESHOLD" if temp_dist[k] >= DIST_THRESHOLD else " ✓"
    print(f"  {pair_labels[k]:30s}: {temp_dist[k]*100:.4f}%{flag}")

print(f"\n── Joint Distinguishability (spectral + temporal) ───────────────")
print(f"Nominal T:")
for k in range(len(pairs)):
    flag = " ← EXCEEDS THRESHOLD" if joint_dist_nom[k] >= DIST_THRESHOLD else " ✓"
    print(f"  {pair_labels[k]:30s}: {joint_dist_nom[k]*100:.3f}%{flag}")
print(f"\nHot extreme (+60°C):")
for k in range(len(pairs)):
    flag = " ← EXCEEDS THRESHOLD" if joint_dist_hot[k] >= DIST_THRESHOLD else " ✓"
    print(f"  {pair_labels[k]:30s}: {joint_dist_hot[k]*100:.3f}%{flag}")

print(f"\n── Stage 1 → 2 → 3 Interface Summary ───────────────────────────")
print(f"  Stage 1 max safe spread:      0.78 nm")
print(f"  Current spread (nominal T):   {max_sep_nom:.3f} nm  "
      f"{'✓ within budget' if max_sep_nom < 0.78 else '✗ exceeds budget'}")
print(f"  Stage 2 QBER floor (nom T):   0.040%  ✓")
print(f"  Stage 3 worst joint D (nom):  {joint_dist_nom.max()*100:.3f}%  "
      f"{'✓' if joint_dist_nom.max() < DIST_THRESHOLD else '✗'}")
print(f"\nOutput saved to: stage3_combined_spectrum.png")
