import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# AM, FM, and PM visualization
#
# Message:
#     m(t) = mu*cos(2*pi*f_m*t)
#
# Carrier:
#     c(t) = A_c*cos(2*pi*f_c*t)
#
# Amplitude modulation:
#     s_AM(t) = A_c[1 + m(t)]cos(2*pi*f_c*t)
#
# Frequency modulation:
#     f_i,FM(t) = f_c + k_f*m(t)
#
#     s_FM(t) = A_c*cos[
#         2*pi*f_c*t + 2*pi*k_f*integral(m(tau) d tau)
#     ]
#
# Phase modulation:
#     s_PM(t) = A_c*cos[2*pi*f_c*t + k_p*m(t)]
# ============================================================


# ------------------------------------------------------------
# 1. Signal parameters
# ------------------------------------------------------------

A_c = 1.0          # Carrier amplitude
f_c = 150.0        # Carrier frequency in Hz

mu = 0.7           # Peak amplitude of normalized message signal
f_m = 3.0          # Message frequency in Hz

duration = 1.0     # Total simulation time in seconds
sampling_rate = 5000  # Samples per second

# Frequency-modulation sensitivity
#
# Units:
#     Hz per unit amplitude of m(t)
#
# Since max|m(t)| = mu, the peak FM frequency deviation is
#
#     Delta_f = k_f*mu
#
k_f = 60.0

# Phase-modulation sensitivity
#
# Units:
#     radians per unit amplitude of m(t)
#
# The maximum PM phase deviation is
#
#     Delta_phi_max = k_p*mu
#
k_p = 2.0


# ------------------------------------------------------------
# 2. Time array
# ------------------------------------------------------------

t = np.linspace(
    0.0,
    duration,
    int(sampling_rate * duration),
    endpoint=False,
)

dt = 1.0 / sampling_rate


# ------------------------------------------------------------
# 3. Message signal
#
# The normalized message signal is
#
#     m(t) = mu*cos(2*pi*f_m*t)
#
# Its peak magnitude is mu.
# ------------------------------------------------------------

message = mu * np.cos(2.0 * np.pi * f_m * t)


# ------------------------------------------------------------
# 4. Unmodulated carrier
# ------------------------------------------------------------

carrier_phase = 2.0 * np.pi * f_c * t

carrier = A_c * np.cos(carrier_phase)


# ============================================================
# 5. Amplitude modulation
#
# In AM, the message changes the amplitude of the carrier:
#
#     A(t) = A_c[1 + m(t)]
#
# Therefore,
#
#     s_AM(t) = A_c[1 + m(t)]cos(2*pi*f_c*t)
# ============================================================

am_signal = A_c * (1.0 + message) * np.cos(carrier_phase)

# Positive and negative envelopes
upper_envelope = A_c * (1.0 + message)
lower_envelope = -A_c * (1.0 + message)


# ============================================================
# 6. Frequency modulation
#
# The defining relation for FM is
#
#     f_i,FM(t) = f_c + k_f*m(t)
#
# The instantaneous phase is obtained by integrating frequency:
#
#     theta_FM(t)
#       = 2*pi*integral[f_i,FM(tau) d tau]
#
#       = 2*pi*f_c*t
#         + 2*pi*k_f*integral[m(tau) d tau]
#
# For
#
#     m(t) = mu*cos(2*pi*f_m*t),
#
# its integral is
#
#     integral[m(tau) d tau]
#       = mu/(2*pi*f_m)*sin(2*pi*f_m*t).
#
# Thus,
#
#     theta_FM(t)
#       = 2*pi*f_c*t
#         + (k_f*mu/f_m)*sin(2*pi*f_m*t).
# ============================================================

peak_frequency_deviation_fm = k_f * mu

beta_fm = peak_frequency_deviation_fm / f_m

instantaneous_frequency_fm = f_c + k_f * message

fm_phase_deviation = (
    beta_fm
    * np.sin(2.0 * np.pi * f_m * t)
)

fm_phase = carrier_phase + fm_phase_deviation

fm_signal = A_c * np.cos(fm_phase)


# ============================================================
# 7. Phase modulation
#
# In PM, the message directly changes the phase:
#
#     Delta_phi_PM(t) = k_p*m(t)
#
# Therefore,
#
#     theta_PM(t) = 2*pi*f_c*t + k_p*m(t)
#
# and
#
#     s_PM(t) = A_c*cos[2*pi*f_c*t + k_p*m(t)].
#
# The corresponding instantaneous frequency is
#
#     f_i,PM(t)
#       = (1/2*pi)*d(theta_PM)/dt
#
#       = f_c + k_p/(2*pi)*dm(t)/dt.
# ============================================================

pm_phase_deviation = k_p * message

pm_phase = carrier_phase + pm_phase_deviation

pm_signal = A_c * np.cos(pm_phase)

# Analytic derivative of
#
#     m(t) = mu*cos(2*pi*f_m*t)
#
message_derivative = (
    -2.0
    * np.pi
    * f_m
    * mu
    * np.sin(2.0 * np.pi * f_m * t)
)

instantaneous_frequency_pm = (
    f_c
    + (k_p / (2.0 * np.pi)) * message_derivative
)

peak_frequency_deviation_pm = k_p * mu * f_m


# ------------------------------------------------------------
# 8. Useful indices for annotations
# ------------------------------------------------------------

message_max_index = np.argmax(message)
message_min_index = np.argmin(message)

selected_time = 0.42
selected_index = np.argmin(np.abs(t - selected_time))


# ============================================================
# Figure 1: Message signal
# ============================================================

plt.figure(figsize=(11, 4))

plt.plot(
    t,
    message,
    linewidth=2,
    label=r"$m(t)$",
)

plt.axhline(0.0, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel(r"$m(t)$")

plt.title(
    r"Message signal: "
    r"$m(t)=\mu\cos(2\pi f_m t)$"
)

plt.annotate(
    rf"Peak message amplitude: $\mu={mu}$",
    xy=(
        t[message_max_index],
        message[message_max_index],
    ),
    xytext=(0.16, 0.82),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 2: Unmodulated carrier
# ============================================================

# Show only a short interval so the carrier cycles are visible.
carrier_display_time = 0.10
carrier_mask = t <= carrier_display_time

plt.figure(figsize=(11, 4))

plt.plot(
    t[carrier_mask],
    carrier[carrier_mask],
    linewidth=1.5,
    label=r"$c(t)$",
)

plt.axhline(0.0, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel(r"$c(t)$")

plt.title(
    r"Unmodulated carrier: "
    r"$c(t)=A_c\cos(2\pi f_c t)$"
)

plt.annotate(
    rf"$A_c={A_c}$, $f_c={f_c}\,\mathrm{{Hz}}$",
    xy=(0.002, A_c),
    xytext=(0.30, 0.82),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 3: Amplitude-modulated signal
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    t,
    am_signal,
    linewidth=1,
    label=r"$s_{\mathrm{AM}}(t)$",
)

plt.plot(
    t,
    upper_envelope,
    linestyle="--",
    linewidth=2,
    label=r"Upper envelope $A_c[1+m(t)]$",
)

plt.plot(
    t,
    lower_envelope,
    linestyle="--",
    linewidth=2,
    label=r"Lower envelope $-A_c[1+m(t)]$",
)

plt.axhline(
    A_c,
    linestyle=":",
    linewidth=1,
    label=r"Unmodulated amplitude $+A_c$",
)

plt.axhline(
    -A_c,
    linestyle=":",
    linewidth=1,
)

plt.axhline(0.0, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    r"Amplitude modulation: "
    r"$s_{\mathrm{AM}}(t)="
    r"A_c[1+m(t)]\cos(2\pi f_c t)$"
)

plt.annotate(
    r"The message controls the envelope",
    xy=(
        t[message_max_index],
        upper_envelope[message_max_index],
    ),
    xytext=(0.19, 0.91),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.annotate(
    r"The carrier frequency remains $f_c$",
    xy=(
        t[selected_index],
        am_signal[selected_index],
    ),
    xytext=(0.57, 0.18),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 4: Frequency-modulated signal
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    t,
    fm_signal,
    linewidth=1,
    label=r"$s_{\mathrm{FM}}(t)$",
)

plt.axhline(
    A_c,
    linestyle=":",
    linewidth=1,
    label=r"Constant amplitude $+A_c$",
)

plt.axhline(
    -A_c,
    linestyle=":",
    linewidth=1,
)

plt.axhline(0.0, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    r"Frequency modulation: "
    r"$s_{\mathrm{FM}}(t)="
    r"A_c\cos\left[2\pi f_c t+"
    r"\beta_{\mathrm{FM}}\sin(2\pi f_m t)\right]$"
)

plt.annotate(
    r"$m(t)>0$: carrier cycles are compressed",
    xy=(
        t[message_max_index],
        fm_signal[message_max_index],
    ),
    xytext=(0.08, 0.88),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.annotate(
    r"$m(t)<0$: carrier cycles are spread out",
    xy=(
        t[message_min_index],
        fm_signal[message_min_index],
    ),
    xytext=(0.58, 0.18),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 5: FM instantaneous frequency
#
# The frequency deviation is
#
#     delta_f(t) = f_i,FM(t) - f_c = k_f*m(t).
# ============================================================

plt.figure(figsize=(12, 4))

plt.plot(
    t,
    instantaneous_frequency_fm,
    linewidth=2,
    label=r"$f_{i,\mathrm{FM}}(t)=f_c+k_fm(t)$",
)

plt.axhline(
    f_c,
    linestyle="--",
    linewidth=1.5,
    label=r"Carrier frequency $f_c$",
)

plt.axhline(
    f_c + peak_frequency_deviation_fm,
    linestyle=":",
    linewidth=1,
    label=r"$f_c+\Delta f$",
)

plt.axhline(
    f_c - peak_frequency_deviation_fm,
    linestyle=":",
    linewidth=1,
    label=r"$f_c-\Delta f$",
)

plt.xlabel("Time (s)")
plt.ylabel("Instantaneous frequency (Hz)")

plt.title(
    r"FM instantaneous frequency: "
    r"$f_{i,\mathrm{FM}}(t)-f_c=k_fm(t)$"
)

plt.annotate(
    rf"$\Delta f=k_f\mu="
    rf"{peak_frequency_deviation_fm:.1f}\,\mathrm{{Hz}}$",
    xy=(
        t[message_max_index],
        instantaneous_frequency_fm[message_max_index],
    ),
    xytext=(0.20, 0.82),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 6: FM phase deviation
#
# Although FM directly controls frequency, the signal itself
# is still generated from a phase:
#
#     Delta_phi_FM(t)
#       = 2*pi*k_f*integral[m(tau) d tau].
# ============================================================

plt.figure(figsize=(12, 4))

plt.plot(
    t,
    fm_phase_deviation,
    linewidth=2,
    label=r"$\Delta\phi_{\mathrm{FM}}(t)$",
)

plt.axhline(0.0, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Phase deviation (rad)")

plt.title(
    r"FM phase deviation: "
    r"$\Delta\phi_{\mathrm{FM}}(t)="
    r"\beta_{\mathrm{FM}}\sin(2\pi f_m t)$"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 7: Phase-modulated signal
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    t,
    pm_signal,
    linewidth=1,
    label=r"$s_{\mathrm{PM}}(t)$",
)

plt.axhline(
    A_c,
    linestyle=":",
    linewidth=1,
    label=r"Constant amplitude $+A_c$",
)

plt.axhline(
    -A_c,
    linestyle=":",
    linewidth=1,
)

plt.axhline(0.0, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    r"Phase modulation: "
    r"$s_{\mathrm{PM}}(t)="
    r"A_c\cos[2\pi f_c t+k_pm(t)]$"
)

plt.annotate(
    r"The message directly controls phase",
    xy=(
        t[message_max_index],
        pm_signal[message_max_index],
    ),
    xytext=(0.18, 0.88),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 8: PM phase deviation
#
# PM defining relation:
#
#     Delta_phi_PM(t) = k_p*m(t).
# ============================================================

plt.figure(figsize=(12, 4))

plt.plot(
    t,
    pm_phase_deviation,
    linewidth=2,
    label=r"$\Delta\phi_{\mathrm{PM}}(t)=k_pm(t)$",
)

plt.axhline(0.0, linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Phase deviation (rad)")

plt.title(
    r"PM phase deviation: "
    r"$\Delta\phi_{\mathrm{PM}}(t)=k_pm(t)$"
)

plt.annotate(
    rf"Maximum phase deviation: "
    rf"$k_p\mu={k_p * mu:.2f}\,\mathrm{{rad}}$",
    xy=(
        t[message_max_index],
        pm_phase_deviation[message_max_index],
    ),
    xytext=(0.20, 0.82),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
)

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 9: PM instantaneous frequency
#
# Since
#
#     theta_PM(t) = 2*pi*f_c*t + k_p*m(t),
#
# differentiation gives
#
#     f_i,PM(t)
#       = f_c + k_p/(2*pi)*dm(t)/dt.
# ============================================================

plt.figure(figsize=(12, 4))

plt.plot(
    t,
    instantaneous_frequency_pm,
    linewidth=2,
    label=(
        r"$f_{i,\mathrm{PM}}(t)="
        r"f_c+\frac{k_p}{2\pi}\frac{dm(t)}{dt}$"
    ),
)

plt.axhline(
    f_c,
    linestyle="--",
    linewidth=1.5,
    label=r"Carrier frequency $f_c$",
)

plt.axhline(
    f_c + peak_frequency_deviation_pm,
    linestyle=":",
    linewidth=1,
    label=r"$f_c+\Delta f_{\mathrm{PM}}$",
)

plt.axhline(
    f_c - peak_frequency_deviation_pm,
    linestyle=":",
    linewidth=1,
    label=r"$f_c-\Delta f_{\mathrm{PM}}$",
)

plt.xlabel("Time (s)")
plt.ylabel("Instantaneous frequency (Hz)")

plt.title(
    r"PM instantaneous frequency: "
    r"$f_{i,\mathrm{PM}}(t)-f_c="
    r"\frac{k_p}{2\pi}\frac{dm(t)}{dt}$"
)

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 10: Direct comparison of AM, FM, and PM
#
# This comparison uses a shorter interval so that differences
# among the waveforms are easier to inspect.
# ============================================================

comparison_end_time = 0.35
comparison_mask = t <= comparison_end_time

fig, axes = plt.subplots(
    4,
    1,
    figsize=(13, 10),
    sharex=True,
)

axes[0].plot(
    t[comparison_mask],
    message[comparison_mask],
    linewidth=2,
)

axes[0].set_ylabel(r"$m(t)$")
axes[0].set_title("Message and modulated signals")
axes[0].grid(True)

axes[1].plot(
    t[comparison_mask],
    am_signal[comparison_mask],
    linewidth=1,
)

axes[1].plot(
    t[comparison_mask],
    upper_envelope[comparison_mask],
    linestyle="--",
    linewidth=1.5,
)

axes[1].plot(
    t[comparison_mask],
    lower_envelope[comparison_mask],
    linestyle="--",
    linewidth=1.5,
)

axes[1].set_ylabel("AM")
axes[1].grid(True)

axes[2].plot(
    t[comparison_mask],
    fm_signal[comparison_mask],
    linewidth=1,
)

axes[2].set_ylabel("FM")
axes[2].grid(True)

axes[3].plot(
    t[comparison_mask],
    pm_signal[comparison_mask],
    linewidth=1,
)

axes[3].set_ylabel("PM")
axes[3].set_xlabel("Time (s)")
axes[3].grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# Print important quantities
# ============================================================

maximum_am_envelope = A_c * (1.0 + mu)
minimum_am_envelope = A_c * (1.0 - mu)

print("=" * 60)
print("COMMON SIGNAL PARAMETERS")
print("=" * 60)

print(f"Carrier amplitude A_c                  = {A_c}")
print(f"Carrier frequency f_c                  = {f_c} Hz")
print(f"Message frequency f_m                  = {f_m} Hz")
print(f"Message peak amplitude mu              = {mu}")

print("\n" + "=" * 60)
print("AMPLITUDE MODULATION")
print("=" * 60)

print(f"Maximum AM envelope A_c(1 + mu)        = {maximum_am_envelope}")
print(f"Minimum AM envelope A_c(1 - mu)        = {minimum_am_envelope}")

if mu < 1.0:
    print("AM regime: under-modulation.")
    print("The envelope does not cross zero.")
elif np.isclose(mu, 1.0):
    print("AM regime: 100% modulation.")
    print("The envelope just touches zero.")
else:
    print("AM regime: overmodulation.")
    print("The envelope crosses zero.")

print("\n" + "=" * 60)
print("FREQUENCY MODULATION")
print("=" * 60)

print(f"FM sensitivity k_f                     = {k_f} Hz/unit")
print(
    f"Peak FM frequency deviation Delta_f    = "
    f"{peak_frequency_deviation_fm} Hz"
)
print(f"FM modulation index beta_FM            = {beta_fm}")
print(
    f"FM instantaneous-frequency range       = "
    f"[{f_c - peak_frequency_deviation_fm}, "
    f"{f_c + peak_frequency_deviation_fm}] Hz"
)

print("\n" + "=" * 60)
print("PHASE MODULATION")
print("=" * 60)

print(f"PM sensitivity k_p                     = {k_p} rad/unit")
print(
    f"Maximum PM phase deviation             = "
    f"{k_p * mu} rad"
)
print(
    f"Peak PM frequency deviation            = "
    f"{peak_frequency_deviation_pm} Hz"
)
print(
    f"PM instantaneous-frequency range       = "
    f"[{f_c - peak_frequency_deviation_pm}, "
    f"{f_c + peak_frequency_deviation_pm}] Hz"
)
