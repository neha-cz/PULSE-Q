"""
Stage 2 (final): PM Fiber Coupling & Polarization Extinction Ratio
Jones calculus — correctly computed in each laser's fiber frame.

Key insight: All four BB84 states behave identically in terms of PER
vs misalignment, because PER only depends on how well the laser's
polarization is aligned to its fiber's slow axis — not on the absolute
lab-frame angle of that axis. The fiber frame approach is correct.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# JONES CALCULUS — FIBER FRAME APPROACH
# ─────────────────────────────────────────────
# Each laser has its own PM fiber whose slow axis is pre-aligned
# to that laser's polarization state. We work in the fiber's own
# coordinate frame, where the slow axis = x, fast axis = y.
#
# In fiber frame:
#   - Input state after inline polarizer = [1, 0] (all on slow axis)
#   - Misalignment θ rotates this: some power leaks onto fast axis
#   - Phase retardance Γ accumulates along fiber (doesn't affect PER
#     if θ=0, but mixes axes when θ≠0 due to interference)

def per_in_fiber_frame(misalign_deg, gamma):
    """
    PER computed in the fiber's own frame.
    misalign_deg: coupling misalignment from slow axis (degrees)
    gamma: birefringent phase retardance (radians)
    Returns PER in dB.
    """
    th = np.deg2rad(misalign_deg)
    # Input in fiber frame: laser polarization projected onto fiber axes
    # Slow axis component: cos(θ), Fast axis component: sin(θ)
    E_slow = np.cos(th)
    E_fast = np.sin(th) * np.exp(1j * gamma)   # phase accumulated on fast axis
    P_slow = abs(E_slow)**2
    P_fast = abs(E_fast)**2
    if P_fast < 1e-20:
        return 60.0
    return 10 * np.log10(P_slow / P_fast)

def per_to_qber(per_db):
    ratio = 10**(per_db / 10.0)
    return 1.0 / (1.0 + ratio)

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────

LASER_LABELS  = ['H  (0°)', 'V  (90°)', 'D  (45°)', 'AD (135°)']
LASER_COLORS  = ['#4fc3f7', '#ef5350', '#66bb6a', '#ffa726']

# PM fiber physical parameters
FIBER_LENGTH_M = 0.5           # m
BEAT_LENGTH_M  = 0.003         # m — PANDA PM fiber at 1280 nm
GAMMA_NOMINAL  = 2*np.pi * FIBER_LENGTH_M / BEAT_LENGTH_M  # rad

# Temperature
T_NOMINAL = 20.0
T_MIN, T_MAX = -20.0, 60.0
N_TEMP = 500
T_range = np.linspace(T_MIN, T_MAX, N_TEMP)

# Initial misalignment per laser at T_nominal (degrees)
# Realistic for careful manual fiber coupling
THETA_0 = np.array([1.0, 1.5, 0.8, 1.2])   # degrees

# Thermal drift: coupling angle and retardance both drift with T
D_THETA_DT = 0.05                            # deg/°C
D_GAMMA_DT = np.pi * FIBER_LENGTH_M         # rad/°C

# Thresholds
PER_THRESHOLD = 20.0    # dB
QBER_BUDGET   = 0.02    # 2% allocation for polarization errors
BB84_LIMIT    = 0.11    # 11% total QBER security boundary

# ─────────────────────────────────────────────
# SWEEP 1: PER and QBER vs misalignment angle
# (All 4 states behave identically — shown overlaid)
# ─────────────────────────────────────────────

theta_deg = np.linspace(0, 10, 1000)
per_vs_theta  = np.zeros((4, len(theta_deg)))
qber_vs_theta = np.zeros((4, len(theta_deg)))

for i in range(4):
    for j, th in enumerate(theta_deg):
        per = per_in_fiber_frame(th, GAMMA_NOMINAL)
        per_vs_theta[i, j]  = per
        qber_vs_theta[i, j] = per_to_qber(per)

# Critical angle: where PER drops below 20 dB
# (same for all 4 states since physics is identical)
critical_idx = np.argmax(per_vs_theta[0] < PER_THRESHOLD)
critical_angle = theta_deg[critical_idx] if per_vs_theta[0, critical_idx] < PER_THRESHOLD else None

# ─────────────────────────────────────────────
# SWEEP 2: PER and QBER vs Temperature
# (lasers differ because they have different θ₀ and drift rates)
# ─────────────────────────────────────────────

per_vs_T  = np.zeros((4, N_TEMP))
qber_vs_T = np.zeros((4, N_TEMP))

for i in range(4):
    for j, T in enumerate(T_range):
        dT      = T - T_NOMINAL
        theta_T = np.clip(THETA_0[i] + D_THETA_DT * dT, 0, 90)
        gamma_T = GAMMA_NOMINAL + D_GAMMA_DT * dT
        per     = per_in_fiber_frame(theta_T, gamma_T)
        per_vs_T[i, j]  = per
        qber_vs_T[i, j] = per_to_qber(per)

total_qber_vs_T = np.mean(qber_vs_T, axis=0)

# ─────────────────────────────────────────────
# SWEEP 3: PER vs retardance (phase sensitivity at fixed θ=2°)
# ─────────────────────────────────────────────

gamma_sweep  = np.linspace(0, 4*np.pi, 1000)
THETA_FIXED  = 2.0  # degrees
per_vs_gamma = np.array([per_in_fiber_frame(THETA_FIXED, g) for g in gamma_sweep])

# Retardance has NO effect on PER (P_slow/P_fast = cos²θ/sin²θ, independent of gamma)
# This is the correct physics result: PER only depends on misalignment angle.

# ─────────────────────────────────────────────
# SWEEP 4: QBER floor budget — how tight must alignment be?
# ─────────────────────────────────────────────

theta_budget  = np.linspace(0, 8, 500)
qber_budget   = np.array([per_to_qber(per_in_fiber_frame(th, GAMMA_NOMINAL))
                           for th in theta_budget])

# Angle at which each QBER threshold is crossed
def angle_at_qber(target_qber):
    idx = np.argmax(qber_budget >= target_qber)
    return theta_budget[idx] if qber_budget[idx] >= target_qber else None

angle_at_2pct  = angle_at_qber(0.02)
angle_at_5pct  = angle_at_qber(0.05)
angle_at_11pct = angle_at_qber(0.11)

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 16), facecolor='#0a0e1a')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35,
                        left=0.07, right=0.97, top=0.93, bottom=0.07)

ACCENT = '#00e5ff'
WARN   = '#ff6b35'
GOOD   = '#00e676'
GRID_C = '#1e2a3a'
TEXT_C = '#cdd6f4'
plt.rcParams.update({'text.color':TEXT_C,'axes.labelcolor':TEXT_C,
                     'xtick.color':TEXT_C,'ytick.color':TEXT_C})

fig.suptitle('Stage 2 — PM Fiber Coupling & Polarization Extinction Ratio\n'
             'Jones Calculus (Fiber Frame) | CubeSat BB84 QKD @ 1280 nm',
             fontsize=16, color=ACCENT, fontweight='bold', y=0.98)

# ── Plot 1: PER vs misalignment ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for i in range(4):
    ax1.plot(theta_deg, per_vs_theta[i], color=LASER_COLORS[i],
             linewidth=2.2, label=LASER_LABELS[i], alpha=0.85)
ax1.axhline(PER_THRESHOLD, color=WARN, linestyle='--', linewidth=1.8,
            label=f'QKD threshold ({PER_THRESHOLD:.0f} dB)')
if critical_angle:
    ax1.axvline(critical_angle, color=WARN, linestyle=':', linewidth=1.5,
                label=f'Critical angle: {critical_angle:.2f}°')
ax1.fill_between(theta_deg, 0, PER_THRESHOLD, alpha=0.08, color=WARN)
ax1.fill_between(theta_deg, PER_THRESHOLD, 65, alpha=0.05, color=GOOD)
ax1.text(0.4, 35, 'SECURE\nREGION', color=GOOD, fontsize=9,
         alpha=0.7, fontweight='bold')
ax1.text(5.0, 8, 'INSECURE\nREGION', color=WARN, fontsize=9,
         alpha=0.7, fontweight='bold')
ax1.set_xlabel('Coupling Misalignment θ (degrees)')
ax1.set_ylabel('PER (dB)')
ax1.set_title('PER vs Coupling Misalignment\n(fiber frame — all 4 states follow same curve)',
              color=TEXT_C, pad=8)
ax1.legend(fontsize=8, framealpha=0.2)
ax1.set_facecolor('#0d1117')
ax1.grid(color=GRID_C, linewidth=0.5)
ax1.set_ylim(0, 65); ax1.set_xlim(0, 10)

# ── Plot 2: QBER contribution vs misalignment ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(theta_budget, qber_budget*100, color=ACCENT, linewidth=2.5,
         label='QBER from polarization error')
ax2.axhline(QBER_BUDGET*100, color=WARN, linestyle='--', linewidth=1.5,
            label=f'Budget: {QBER_BUDGET*100:.0f}%')
ax2.axhline(BB84_LIMIT*100, color='#ff1744', linestyle='--', linewidth=1.5,
            label=f'BB84 limit: {BB84_LIMIT*100:.0f}%')
if angle_at_2pct:
    ax2.axvline(angle_at_2pct, color=WARN, linestyle=':', linewidth=1.3,
                label=f'2% budget → θ < {angle_at_2pct:.2f}°')
if angle_at_11pct:
    ax2.axvline(angle_at_11pct, color='#ff1744', linestyle=':', linewidth=1.3,
                label=f'11% limit → θ < {angle_at_11pct:.2f}°')
ax2.fill_between(theta_budget, 0, QBER_BUDGET*100, alpha=0.10, color=GOOD)
ax2.fill_between(theta_budget, QBER_BUDGET*100, BB84_LIMIT*100,
                 alpha=0.07, color=WARN)
ax2.fill_between(theta_budget, BB84_LIMIT*100, 15, alpha=0.07, color='#ff1744')
ax2.set_xlabel('Coupling Misalignment θ (degrees)')
ax2.set_ylabel('QBER Contribution (%)')
ax2.set_title('QBER Contribution from Polarization Error\n(alignment tolerance budget)',
              color=TEXT_C, pad=8)
ax2.legend(fontsize=8, framealpha=0.2)
ax2.set_facecolor('#0d1117')
ax2.grid(color=GRID_C, linewidth=0.5)
ax2.set_ylim(0, 15); ax2.set_xlim(0, 8)

# ── Plot 3: PER vs Temperature (per laser, different θ₀) ──────────────────
ax3 = fig.add_subplot(gs[1, 0])
for i in range(4):
    ax3.plot(T_range, per_vs_T[i], color=LASER_COLORS[i],
             linewidth=2.2, label=f'{LASER_LABELS[i]}  (θ₀={THETA_0[i]}°)')
ax3.axhline(PER_THRESHOLD, color=WARN, linestyle='--', linewidth=1.8,
            label=f'QKD threshold ({PER_THRESHOLD:.0f} dB)')
ax3.axvline(T_NOMINAL, color='white', linestyle='--', alpha=0.3, linewidth=1,
            label='Nominal T (20°C)')
ax3.fill_between(T_range, 0, PER_THRESHOLD, alpha=0.08, color=WARN)
ax3.set_xlabel('Temperature (°C)')
ax3.set_ylabel('PER (dB)')
ax3.set_title('PER vs Orbital Temperature\n(lasers differ due to individual θ₀ and drift)',
              color=TEXT_C, pad=8)
ax3.legend(fontsize=8, framealpha=0.2)
ax3.set_facecolor('#0d1117')
ax3.grid(color=GRID_C, linewidth=0.5)
ax3.set_ylim(0, 65)

# ── Plot 4: QBER floor vs Temperature ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for i in range(4):
    ax4.plot(T_range, qber_vs_T[i]*100, color=LASER_COLORS[i],
             linewidth=1.5, alpha=0.55, label=LASER_LABELS[i])
ax4.plot(T_range, total_qber_vs_T*100, color='white',
         linewidth=2.8, label='Total QBER floor (mean)', zorder=5)
ax4.axhline(QBER_BUDGET*100, color=WARN, linestyle='--', linewidth=1.5,
            label=f'QBER budget ({QBER_BUDGET*100:.0f}%)')
ax4.axhline(BB84_LIMIT*100, color='#ff1744', linestyle='--', linewidth=1.5,
            label=f'BB84 limit ({BB84_LIMIT*100:.0f}%)')
ax4.axvline(T_NOMINAL, color='white', linestyle='--', alpha=0.3, linewidth=1)
ax4.fill_between(T_range, 0, QBER_BUDGET*100, alpha=0.08, color=GOOD)
ax4.set_xlabel('Temperature (°C)')
ax4.set_ylabel('QBER Contribution (%)')
ax4.set_title('Polarization QBER Floor vs Orbital Temperature',
              color=TEXT_C, pad=8)
ax4.legend(fontsize=8, framealpha=0.2, ncol=2)
ax4.set_facecolor('#0d1117')
ax4.grid(color=GRID_C, linewidth=0.5)

# ── Plot 5: PER vs retardance (confirm independence) ──────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(gamma_sweep/np.pi, per_vs_gamma, color=ACCENT, linewidth=2.5,
         label=f'PER at θ = {THETA_FIXED}°')
ax5.axhline(PER_THRESHOLD, color=WARN, linestyle='--', linewidth=1.5,
            label=f'QKD threshold ({PER_THRESHOLD:.0f} dB)')
ax5.axhline(per_vs_gamma[0], color=GOOD, linestyle=':', linewidth=1.2, alpha=0.7,
            label=f'PER = {per_vs_gamma[0]:.1f} dB (flat — retardance independent)')
ax5.set_xlabel('Fiber Phase Retardance Γ (units of π rad)')
ax5.set_ylabel('PER (dB)')
ax5.set_title(f'PER vs Fiber Retardance (θ = {THETA_FIXED}°)\n'
              'Flat response confirms PER depends only on misalignment, not Γ',
              color=TEXT_C, pad=8)
ax5.legend(fontsize=8, framealpha=0.2)
ax5.set_facecolor('#0d1117')
ax5.grid(color=GRID_C, linewidth=0.5)
ax5.set_ylim(0, 65)

# ── Plot 6: Axis power split — tolerance guide ────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
theta_fine  = np.linspace(0, 10, 500)
wrong_pct   = np.sin(np.deg2rad(theta_fine))**2 * 100
correct_pct = np.cos(np.deg2rad(theta_fine))**2 * 100

ax6.plot(theta_fine, correct_pct, color=GOOD, linewidth=2.5,
         label='Correct axis (cos²θ)')
ax6.plot(theta_fine, wrong_pct, color=WARN, linewidth=2.5,
         label='Wrong axis leakage (sin²θ)')
ax6.fill_between(theta_fine, 0, wrong_pct, alpha=0.15, color=WARN)

for pct, col, lbl in [(1.0,'#ffeb3b','1% leakage'),
                       (2.0, WARN,    '2% budget'),
                       (5.0,'#ff1744','5% leakage')]:
    th = np.rad2deg(np.arcsin(np.sqrt(pct/100)))
    ax6.axvline(th, color=col, linestyle=':', linewidth=1.3, alpha=0.9,
                label=f'{lbl} → θ = {th:.2f}°')

ax6.set_xlabel('Coupling Misalignment θ (degrees)')
ax6.set_ylabel('Power on Axis (%)')
ax6.set_title('Slow/Fast Axis Power Split vs Misalignment\n'
              'Alignment tolerance procurement guide (sin²θ / cos²θ)',
              color=TEXT_C, pad=8)
ax6.legend(fontsize=8, framealpha=0.2)
ax6.set_facecolor('#0d1117')
ax6.grid(color=GRID_C, linewidth=0.5)
ax6.set_ylim(0, 12)

plt.savefig('/mnt/user-data/outputs/stage2_jones_per.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("=" * 62)
print("STAGE 2 SUMMARY — PM Fiber Coupling & PER (final)")
print("=" * 62)
print(f"\nFiber length:             {FIBER_LENGTH_M} m")
print(f"Beat length (PANDA):      {BEAT_LENGTH_M*1000:.1f} mm")
print(f"Nominal retardance:       {GAMMA_NOMINAL/np.pi:.1f}π rad")
print(f"\nKey physics result:")
print(f"  PER depends ONLY on misalignment angle θ, NOT on")
print(f"  fiber retardance Γ. All 4 BB84 states behave identically.")
print(f"\nCritical misalignment angle (PER = {PER_THRESHOLD} dB):")
print(f"  θ_critical = {critical_angle:.2f}° (same for all 4 states)")

print(f"\nQBER budget alignment tolerances:")
if angle_at_2pct:
    print(f"  Keep θ < {angle_at_2pct:.2f}° for QBER < 2% from polarization error")
if angle_at_5pct:
    print(f"  Keep θ < {angle_at_5pct:.2f}° for QBER < 5% from polarization error")
if angle_at_11pct:
    print(f"  Keep θ < {angle_at_11pct:.2f}° for QBER < 11% (BB84 hard limit)")

print(f"\nInitial state (T = {T_NOMINAL}°C):")
tidx_nom = np.argmin(np.abs(T_range - T_NOMINAL))
for i in range(4):
    per  = per_vs_T[i, tidx_nom]
    qber = qber_vs_T[i, tidx_nom]
    flag = "  ← BELOW PER THRESHOLD" if per < PER_THRESHOLD else ""
    print(f"  {LASER_LABELS[i]:12s}: θ₀={THETA_0[i]:.1f}°  "
          f"PER={per:.1f} dB  QBER_contribution={qber*100:.4f}%{flag}")
print(f"  Total QBER floor: {total_qber_vs_T[tidx_nom]*100:.4f}%")

for T_check, Tname in [(-20,'Cold (-20°C)'), (60,'Hot (+60°C)')]:
    tidx = np.argmin(np.abs(T_range - T_check))
    print(f"\n{Tname}:")
    for i in range(4):
        per  = per_vs_T[i, tidx]
        qber = qber_vs_T[i, tidx]
        flag = "  ← BELOW THRESHOLD" if per < PER_THRESHOLD else ""
        print(f"  {LASER_LABELS[i]:12s}: PER={per:.1f} dB  "
              f"QBER_contribution={qber*100:.4f}%{flag}")
    print(f"  Total QBER floor: {total_qber_vs_T[tidx]*100:.4f}%")

print("\n─── Interface to Stage 3 / Protocol Layer ────────────────")
tidx_hot = np.argmin(np.abs(T_range - 60))
print(f"  From Stage 1: max safe wavelength spread = 0.78 nm")
print(f"  From Stage 2: QBER floor (nominal T)     = "
      f"{total_qber_vs_T[tidx_nom]*100:.4f}%")
print(f"  From Stage 2: QBER floor (hot extreme)   = "
      f"{total_qber_vs_T[tidx_hot]*100:.4f}%")
print(f"  From Stage 2: alignment tolerance        = "
      f"< {critical_angle:.2f}° per laser coupling")
print("\nOutput saved to: stage2_jones_per.png")
