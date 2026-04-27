import numpy as np
import matplotlib.pyplot as plt

# =========================
# Operator
# =========================

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

s_plus  = (sx + 1j * sy)/2
s_minus = (sx - 1j * sy)/2

# =========================
# Fixed axis & variable state
# =========================

def ground_state(theta):
    return np.array([
        np.cos(theta/2),
        np.sin(theta/2)
    ], dtype=complex)

def excited_state(theta):
    return np.array([
        -np.sin(theta/2),
        np.cos(theta/2)
    ], dtype=complex)

# =========================
# ESR intensity
# =========================

def transition_intensity(theta):
    g = ground_state(theta)
    e = excited_state(theta)
    
    I_plus  = np.abs(np.vdot(e, s_plus @ g))**2
    I_minus = np.abs(np.vdot(e, s_minus @ g))**2
    
    # linear polarization
    I_x = np.abs(np.vdot(e, sx @ g))**2
    I_y = np.abs(np.vdot(e, sy @ g))**2
    
    return I_plus, I_minus, I_x, I_y

# =========================
# θ reconstruction
# =========================

def extract_theta(Ip, Im, Ix, Iy):
    pC = (Ip - Im) / (Ip + Im)
    pL = (Ix - Iy) / (Ix + Iy)
    
    theta = np.arctan2(pL, pC)
    return theta

# =========================
# QGT
# =========================

def compute_metric(theta_vals, theta_rec):
    theta_rec = np.unwrap(theta_rec)
    dtheta = np.gradient(theta_rec, theta_vals)
    G = 0.25 * dtheta**2
    return G

# =========================
# Simulation
# =========================

theta_vals = np.linspace(0.01, 2*np.pi-0.01, 400)

I_plus_list = []
I_minus_list = []
theta_rec_list = []

for th in theta_vals:
    Ip, Im, Ix, Iy = transition_intensity(th)
    theta_rec = extract_theta(Ip, Im, Ix, Iy)
    
    I_plus_list.append(Ip)
    I_minus_list.append(Im)
    theta_rec_list.append(theta_rec)

I_plus_list = np.array(I_plus_list)
I_minus_list = np.array(I_minus_list)
theta_rec_list = np.array(theta_rec_list)

G_vals = compute_metric(theta_vals, theta_rec_list)

# =========================
# Theoretical value
# =========================

G_exact = 0.25 * np.ones_like(theta_vals)

# =========================
# Plot
# =========================

title_opts  = dict(fontsize=10, pad=6)
label_opts  = dict(fontsize=9)
legend_opts = dict(fontsize=8, frameon=False)

plt.figure(figsize=(12, 9))

# ESR intensity
plt.subplot(2,2,1)
plt.plot(theta_vals, I_plus_list, label="I+", linewidth=1.5)
plt.plot(theta_vals, I_minus_list, label="I-", linewidth=1.5)

plt.xticks(
    [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
)

plt.title("ESR Intensities (Correct θ dependence)", **title_opts)
plt.xlabel("θ (rad)", **label_opts)
plt.ylabel("Intensity (arb. units)", **label_opts)
plt.legend(loc="best", **legend_opts)

# θ reconstruction
plt.subplot(2,2,2)
plt.plot(theta_vals, theta_vals, '--', label="True θ", linewidth=1.5)
plt.plot(theta_vals, np.unwrap(theta_rec_list), label="Reconstructed θ", linewidth=1.5)

plt.xticks(
    [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
)
plt.yticks(
    [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
)

plt.title("Theta Reconstruction", **title_opts)
plt.xlabel("θ", **label_opts)
plt.ylabel("θ", **label_opts)
plt.legend(loc="best", **legend_opts)

# Quantum Metric
plt.subplot(2,2,3)
plt.plot(theta_vals, G_vals, label="Reconstructed G", linewidth=1.5)
plt.plot(theta_vals, G_exact, '--', label="Exact G = 1/4", linewidth=1.5)

plt.xticks(
    [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
)

plt.title("Quantum Metric", **title_opts)
plt.xlabel("θ", **label_opts)
plt.ylabel("G", **label_opts)
plt.legend(loc="best", **legend_opts)

# Circular Dichroism
plt.subplot(2,2,4)
pC = (I_plus_list - I_minus_list) / (I_plus_list + I_minus_list)
plt.plot(theta_vals, pC, linewidth=1.5)

plt.xticks(
    [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
)

plt.title("Circular Dichroism", **title_opts)
plt.xlabel("θ", **label_opts)

plt.tick_params(labelsize=8)
plt.tight_layout(pad=1.0)
plt.show()
