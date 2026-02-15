import matplotlib.pyplot as plt
import numpy as np


def schwarzschild_radius(M):
    """
    Calcul du rayon de Schwarzschild pour une masse M.
    r_s = 2GM/c^2
    """
    G = 6.67430e-11  # Constante gravitationnelle en m^3 kg^-1 s^-2
    c = 299792458     # Vitesse de la lumière en m/s
    return 2 * G * M / c**2

c = 299792458  # Vitesse de la lumière en m/s
G=6.67430e-11  # Constante gravitationnelle en m^3 kg^-1 s^-2
Msun = 1.989e30  # Masse du Soleil en kg    
def ff_EF_time(r0, r, M):
    """
    Calcul du temps coordonné par intégration numérique.
    r décroît de r0 vers 0, donc on separe entre les deux régions:
- r > r_s : descente vers le trou noir
- r < r_s : chute libre à l'intérieur du trou noir
    dt/dr = sqrt(1-r_s/r0) / (1-r_s/r) * (-1/(c*sqrt(r_s/r - r_s/r0)))
    """
    r_s = schwarzschild_radius(M)
    r = np.asarray(r)
    dt_dr = np.zeros_like(r, dtype=float)
    
    # Masques pour les deux régions
    mask_out = r > r_s  # Région r > r_s (reachable)
    mask_in = r < r_s   # Région r < r_s (intérieur)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Région extérieure
        if np.any(mask_out):
            dt_dr[mask_out] = np.sqrt(1 - r_s/r0) / (1 - r_s/r[mask_out]) * (-1) / (c * np.sqrt(r_s/r[mask_out] - r_s/r0)) + 1/c * 1/(r[mask_out]/r_s - 1)
        
        # Région intérieure
        if np.any(mask_in):
            dt_dr[mask_in] = np.sqrt(1 - r_s/r0) / (1 - r_s/r[mask_in]) * (-1) / (c * np.sqrt(r_s/r[mask_in] - r_s/r0)) + 1/c * 1/(r[mask_in]/r_s - 1)
            #dt_dr[mask_in] = np.clip(dt_dr[mask_in], -2, None)  # Clip à -2 pour régulariser à r_s
    
    dr = np.gradient(r)
    t_coord = np.cumsum((dt_dr * dr))
    t_coord += 20*r_s/c  # Normaliser pour que t=0 à r=r0
    return t_coord

fig, ax = plt.subplots()
M = 10 * Msun  # Masse du trou noir en kg
r_s = schwarzschild_radius(M)
r0 = 5 * r_s  # Rayon initial de chute libre

r = np.linspace(0.1 * schwarzschild_radius(M), 5 * schwarzschild_radius(M), 1000)  # Rayon de chute libre
t_coord = ff_EF_time(r0, r, M)
ax.plot(r/r_s, t_coord/r_s*c, label='Temps coordonné (EF)')
ax.set_xlabel('Rayon (r/r_s)')
ax.set_ylabel('Temps coordonné EF / (r_s/c)')
ax.set_title('Temps coordonné en fonction du rayon pour une chute libre')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
fig.savefig('temps_coordonne.png', dpi=150, bbox_inches='tight')