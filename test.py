import numpy as np
import matplotlib.pyplot as plt

# --- Paramètres physiques (Unités : 2M = 1) ---
M = 0.5
rs = 1.0  # Rayon de Schwarzschild
Ri = 10.0 * M  # Rayon initial (5.0 dans ces unités)
dt_shift = 44 * M  # Décalage temporel spécifié dans le document

def generate_kruskal_path():
    # 1. Paramétrage cycloïdal (eta de 0 à pi)
    eta = np.linspace(0, np.pi - 0.001, 2000)
    r = (Ri / 2) * (1 + np.cos(eta))
    
    # 2. Calcul des termes de temps (MTW 31.10)
    k = np.sqrt(Ri/rs - 1)
    # Partie régulière de t/rs
    t_reg = k * (eta + (Ri/(2*rs)) * (eta + np.sin(eta)))
    
    # 3. Calcul de V+U et V-U pour éviter la divergence
    # V+U = exp((t + r*)/2rs)
    # V-U = +/- exp(-(t - r*)/2rs)
    # On gère analytiquement le produit exp(log|r-rs|) * sqrt|r-rs|
    
    # Terme de l'horizon : tan(eta/2) vaut k quand r = rs
    # On utilise la simplification : sqrt(r/rs - 1) * exp(t/2rs)
    # qui reste finie à l'horizon.
    
    # t_log = rs * ln |(k + tan)/(k - tan)|
    # r_star_log = rs * ln |r/rs - 1|
    
    # Le terme combiné (t + r*)/(2rs) sans les logs divergents :
    # On calcule l'argument de l'exponentielle de manière stable
    exp_arg_plus = (t_reg + r/rs) / 2.0
    
    # Facteur correcteur pour les logs :
    # sqrt|r/rs-1| * exp(0.5 * ln|(k+tan)/(k-tan)|)
    with np.errstate(divide='ignore', invalid='ignore'):
        term_corr = np.sqrt(np.abs(r/rs - 1)) * np.sqrt(np.abs((k + np.tan(eta/2)) / (k - np.tan(eta/2))))
    
    # Coordonnées de Kruskal de base (non boostées)
    # On multiplie par exp(r/2rs) qui vient de r*
    A = np.exp(exp_arg_plus) * term_corr * np.exp(r/(2*rs)) # C'est V+U
    B = (r/rs - 1) * np.exp(r/rs) / A # C'est V-U via la relation métrique
    
    U_base = (A - B) / 2
    V_base = (A + B) / 2
    
    # 4. Application de la translation temporelle (t -> t + 42.8M)
    # Cela correspond à une transformation de Lorentz (boost) dans le plan (U, V)
    alpha = dt_shift / (2 * rs)
    U = U_base * np.cosh(alpha) - V_base * np.sinh(alpha)
    V = V_base * np.cosh(alpha) - U_base * np.sinh(alpha)
    
    return U, V

# --- Tracé du graphique ---
U, V = generate_kruskal_path()

fig, ax = plt.subplots(figsize=(8, 8))

# Horizon (r=2M)
ax.plot([0, 5], [0, 5], 'k--', lw=1, alpha=0.5, label="Horizon (r=2M)")
ax.plot([0, 5], [0, -5], 'k--', lw=1, alpha=0.5)

# Singularité (r=0) -> V^2 - U^2 = 1
u_s = np.linspace(-5, 5, 1000)
v_s = np.sqrt(1 + u_s**2)
ax.plot(u_s, v_s, 'r', lw=2, label="Singularité (r=0)")

# Surface de l'étoile
ax.plot(U, V, 'b', lw=3, label="Surface de l'étoile")
ax.fill_betweenx(V, -5, U, color='gray', alpha=0.2) # Zone intérieure

# Échelles identiques à l'image (b)
ax.set_xlim(0, 4)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_title('Effondrement Stellaire : Kruskal-Szekeres (MTW Fig 32.1 b)')
ax.legend()
ax.grid(True, alpha=0.2)

plt.show()