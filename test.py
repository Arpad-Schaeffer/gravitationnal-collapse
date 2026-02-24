import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Paramètres
M = 1.0
R0 = 4.0 * M 
E = np.sqrt(1 - 2*M/R0)

def surface_evolution_stable():
    # 1. On intègre r et V (Eddington-Finkelstein) par rapport à tau
    # V = t + r* est régulier à l'horizon pour une chute vers l'intérieur.
    def derivatives(tau, y):
        r, V = y
        if r <= 0.01: return [0, 0]
        
        dr_dtau = -np.sqrt(2*M/r - 2*M/R0)
        # Formule stable pour dV/dtau (pas de division par 0 à l'horizon !)
        dV_dtau = 1.0 / (E - dr_dtau)
        
        return [dr_dtau, dV_dtau]

    # Conditions initiales à tau = 0 (on part de t=0)
    # r_star = r + 2M * ln|r/2M - 1|
    r_start = R0 - 1e-6
    V0 = r_start + 2*M*np.log(r_start/(2*M) - 1.0)
    
    sol = solve_ivp(derivatives, (0, 20), [r_start, V0], 
                    t_eval=np.linspace(0, 15, 1000), method='RK45')
    
    r_path = sol.y[0]
    V_path = sol.y[1]

    # 2. CONVERSION STABLE VERS u, v (Sans passer par t !)
    # On utilise les définitions : 
    # v + u = exp(V/4M)
    # v - u = (1 - r/2M) * exp(r/2M) * exp(-V/4M)
    
    v_plus_u = np.exp(V_path / (4.0 * M))
    v_minus_u = (1.0 - r_path/(2.0 * M)) * np.exp(r_path/(2.0 * M)) / v_plus_u
    
    v_k = 0.5 * (v_plus_u + v_minus_u)
    u_k = 0.5 * (v_plus_u - v_minus_u)
    
    # On peut reconstruire t a posteriori pour le graph de gauche
    # t = V - r_star
    r_star = r_path + 2*M*np.log(np.abs(r_path/(2*M) - 1.0))
    t_path = V_path - r_star
    
    return u_k, v_k, r_path, t_path

def plot_collapse():
    u_surf, v_surf, r_path, t_path = surface_evolution_stable()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- GRAPH 1 : SCHWARZSCHILD ---
    axes[0].plot(r_path, t_path, 'brown', lw=3, label="Surface")
    axes[0].axvline(2*M, color='r', ls='--', label="Horizon")
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('t')
    axes[0].set_title('Vue Schwarzschild (gelée à l\'horizon)')
    axes[0].set_ylim(0, 20)
    axes[0].legend()

    # --- GRAPH 2 : KRUSKAL ---
    # Singularité r=0
    u_sing = np.linspace(-4, 4, 500)
    axes[1].plot(u_sing, np.sqrt(1 + u_sing**2), 'k', lw=3, label="Singularité r=0")
    
    # Horizons
    axes[1].plot([-4, 4], [-4, 4], 'k--', alpha=0.3)
    axes[1].plot([-4, 4], [4, -4], 'k--', alpha=0.3)
    
    # Trajectoire de la surface
    axes[1].plot(u_surf, v_surf, 'brown', lw=4, label="Surface de l'étoile")
    
    # Ajout d'un cône de lumière sur la surface pour vérifier du/dv=1
    idx = 500 # milieu de la chute
    u0, v0 = u_surf[idx], v_surf[idx]
    axes[1].plot([u0-0.3, u0+0.3], [v0-0.3, v0+0.3], 'orange', lw=2)
    axes[1].plot([u0-0.3, u0+0.3], [v0+0.3, v0-0.3], 'orange', lw=2)

    axes[1].set_xlim(-1, 4)
    axes[1].set_ylim(-1, 4)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('u')
    axes[1].set_ylabel('v')
    axes[1].set_title('Vue Kruskal (Traversée réelle)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_collapse()