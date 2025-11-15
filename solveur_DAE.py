import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Utilitaires

EPS = 1e-12

# Fonction pour éviter de diviser par 0 et avoir des erreurs par exemple qaund on a un ratio 0/0 indéfini il devient 0/EPS ce qui donne 0.


def clamp_den(x, eps=EPS):
    return x if abs(x) > eps else (np.sign(x) * eps if x != 0 else eps)

# Définition


def d_dx_central(arr, i, dx):
    """Dérivée spatiale centrale ∂/∂x (arr indexé en x), avec garde simple aux bords."""
    if i == 0:
        return (arr[1] - arr[0]) / dx
    if i == len(arr) - 1:
        return (arr[i] - arr[i-1]) / dx
    return (arr[i+1] - arr[i-1]) / (2.0 * dx)


# COUCHE LIMITE (mécanique/flux)

# Formules issues de l'article de Tsuji et Nagano, qui donne un modèle empirique à partir d'observations de flux convectifs

def Ub(g, beta, Tw, TB, nu):
    """Vitesse caractéristique de couche limite (profil intégral)."""
    return (g * beta * (abs(Tw - TB)) * nu) ** (1/3)


def Grashof_x(g, beta, Tw, TB, x, nu):
    """Grashof local Gr_x."""
    return (g * beta * (abs(Tw - TB)) * x**3) / (nu**2)


def tau_w_lam(rho, ub, g, beta, Tw, TB, x, nu):
    """Contrainte pariétale laminaire (Tsuji–Nagano, Gr < 1e9)."""
    Grx = Grashof_x(g, beta, Tw, TB, x, nu)
    return rho * (ub**2) * 0.953 * (Grx ** (1/12))


def tau_w_turb(rho, ub, g, beta, Tw, TB, x, nu):
    """Contrainte pariétale turbulente (Tsuji–Nagano, Gr > 1e9)."""
    Grx = Grashof_x(g, beta, Tw, TB, x, nu)
    return rho * (ub**2) * 0.684 * (Grx ** (1/11.9))

# Equation 18


def Rayleigh_x(g, alpha, Tw, TB, x, nu, a):
    """Ra_x = g α (Tw-TB) x^3 / (ν a)."""
    return (g * alpha * (abs(Tw - TB)) * x**3) / (nu * a)

# Equation 17


def k_nat_1e5_1e9(Rax, lambda_l, x):
    """k pour 1e5 < Ra < 1e9 : Nu = 0.59 Ra^(1/4)."""
    return 0.59 * (Rax ** 0.25) * (lambda_l / clamp_den(x))


def k_nat_1e9_1e13(Rax, lambda_l, x):
    """k pour 1e9 < Ra < 1e13 : Nu = 0.11 Ra^(1/3)."""
    return 0.11 * (Rax ** (1/3)) * (lambda_l / clamp_den(x))


# Equation 16

def q_in_wall(Tw, TB, Rax, x, lambda_l):
    """
    Flux pariétal entrant q'' = k (Tw - TB), avec sélection de corrélation selon Ra_x.
    Hors plage, on prolonge prudemment la corrélation 1e5–1e9.
    """
    dT = (Tw - TB)
    if 1e5 < Rax <= 1e9:
        k = k_nat_1e5_1e9(Rax, lambda_l, x)
    elif 1e9 < Rax <= 1e13:
        k = k_nat_1e9_1e13(Rax, lambda_l, x)
    else:
        k = k_nat_1e5_1e9(max(Rax, 1e5), lambda_l, max(x, EPS))
    return k * dT

# Valeurs déterminées grâce à des lors empirique peut être que des valeurs tabulées seraient meilleures


def k_propane_liq(T):
    """Conductivité thermique [W/m/K] du propane liquide (230–320 K)."""
    return max(0.146 - 1.8e-4 * T, 0.05)


def k_propane_vap(T):
    """Conductivité thermique [W/m/K] du propane vapeur (250–350 K)."""
    return 7.2e-3+3.5e-5*T

# Variables normalisées dans la CL : η


# Equation 19
def eta_moy(T_tilde, TB, Tw):
    """η = (T̃ - TB)/(Tw - TB)."""
    return (T_tilde - TB) / clamp_den(Tw - TB)

# Equation 20


def d_eta_dt(i, dTtilde_dt, dTw_dt, dTB_dt, Tw, TB, eta_i, ddelta_dt):
    """
    dη_i/dt — si dδ/dt >= 0 : 0
             sinon : ( dT̃/dt - η dTw/dt + (1-η) dTB/dt ) / (Tw - TB)
    """
    if ddelta_dt >= 0.0:
        return 0.0
    denom = clamp_den(Tw[i] - TB[i])
    return (dTtilde_dt - eta_i * dTw_dt + (1.0 - eta_i) * dTB_dt) / denom

# Équations intégrales de la couche limite (énergie)


# Equation 21

def dTtilde_dt(i, eta, dTw_dt, dTB_dt,
               Tw, TB,
               ddelta_dt,
               m_dot, m_dot_y, h_tilde, h_y,
               q_in_i, R, Hcv, rho, cv, delta_i):
    if ddelta_dt >= 0.0:
        return eta[i] * dTw_dt + (1.0 - eta[i]) * dTB_dt

    num = (m_dot[i] * h_tilde[i]
           - m_dot[i-1] * h_tilde[i-1]
           - m_dot_y[i] * h_y[i]
           + 2.0 * np.pi * R * Hcv * q_in_i)
    den = 2.0 * np.pi * R * rho * cv * Hcv * clamp_den(delta_i)
    return num / den


# Equation 15


def ddelta_dt_from_energy(i,
                          dTtilde_dt_i,                # valeur scalaire déjà calculée
                          m_dot, m_dot_y, h_tilde, h_y,
                          q_in_i, R, Hcv, rho, cv,
                          delta_i, T_tilde_i, TB_i):
    m_terms = (m_dot[i] * h_tilde[i]
               - m_dot[i-1] * h_tilde[i-1]
               - m_dot_y[i] * h_y[i])
    A = (m_terms / (2.0 * np.pi * R * Hcv * rho * cv)) - \
        (q_in_i / (rho * cv)) - delta_i * dTtilde_dt_i
    return A / clamp_den(T_tilde_i - TB_i)


# Masse/Énergie dans le BULK (liquide)


# Equation 22


def bilan_masse_bulk(mB, m_dot_y_i):
    """Équilibre masse local (retourne mB_i - ṁ_y,i) si besoin d’un résidu."""
    return mB - m_dot_y_i


# Equation 25


def dTB_dt_bulk(i, lambdaB, R, rho, cv,
                TB, delta, dx,
                m_dot_B, m_dot_y, h_y, h_B,
                T_top=None):
    # force 1D float
    TB = np.asarray(TB,      dtype=float).reshape(-1)
    delta = np.asarray(delta,   dtype=float).reshape(-1)
    m_dot_B = np.asarray(m_dot_B, dtype=float).reshape(-1)
    m_dot_y = np.asarray(m_dot_y, dtype=float).reshape(-1)
    h_y = np.asarray(h_y,     dtype=float).reshape(-1)
    h_B = np.asarray(h_B,     dtype=float).reshape(-1)

    N = len(TB)
    area_i = float(np.pi * (R - delta[i])**2)

    # dTB/dx au point i
    if (i == N-1) and (T_top is not None):
        dTBdx_i = float((TB[i] - float(T_top)) / dx)
    else:
        dTBdx_i = float(d_dx_central(TB, i, dx))
    dTBdx_im1 = float(d_dx_central(TB, max(i-1, 0), dx))

    # numérateur (scalaires)
    num = (float(m_dot_B[i]) * float(h_B[i])
           - float(m_dot_B[i-1]) * float(h_B[i-1])
           - float(m_dot_y[i]) * float(h_y[i])
           + area_i * float(lambdaB) * (dTBdx_i - dTBdx_im1))
    den = area_i * float(rho) * float(cv) * float(dx)

    return num / clamp_den(den)


# Interface liquide/vapeur (échanges)

# Equation 18

def Rayleigh_v_int(g, alpha_v, Tv, Tint, D, nu_v, rho_v, cp_v, lambda_vapeur):
    return (g * alpha_v * (abs(Tint - Tv)) * D**3*rho_v*cp_v) / (nu_v * lambda_vapeur)


def Rayleigh_l_int(g, alpha_l, Ts, Tint, D, nu_l, a_l):
    return (g * alpha_l * (abs(Tint - Ts)) * D**3) / (nu_l * a_l)


# Equation 32


def k_v_int(Ra_v_int, D, lambda_v):
    """k côté vapeur à l’interface."""
    return 0.54 * (Ra_v_int ** 0.25) * (lambda_v / clamp_den(D))


def k_l_int(Ra_l_int, D, lambda_l):
    """k côté liquide à l’interface."""
    return 0.27 * (Ra_l_int ** 0.25) * (lambda_l / clamp_den(D))

# Equation 31


def m_dot_phase(R, gamma, Tv, Ts, Tint, k_v_int_val, k_l_int_val):
    return np.pi * R**2 * (k_v_int_val * (Tv - Tint) - k_l_int_val * (Tint - Ts)) / clamp_den(gamma)


# Equation 30


def dHs_dt(m_dot_ph, rho_l, R):
    return m_dot_ph / (rho_l * np.pi * R**2)

# Equation 29


def dTs_dt_surface(n,
                   m_dot_B, h_B,           # tableaux sur x
                   m_dot, h_tilde,         # CL en haut (ṁ_n, h̃_n)
                   R, H_s, qin_s,          # géométrie/surface & flux paroi -> surface
                   kint_liq,               # k_int->liq (interface -> liquide)
                   T_int, T_s,             # températures interface & surface
                   # bulk: épaisseur CL (delta), λ_B, T_B(x)
                   delta, lambda_B, T_B,
                   dx,                     # pas spatial pour dT_B/dx
                   dH_s_dt,                # déjà calculé via dHs_dt(...)
                   rho_liq, cv_liq  # AJOUT : propriétés du liquide
                   ):

    area_surf = np.pi * R**2
    area_bulk = np.pi * (R - delta[n])**2
    H_eff = max(H_s, 1e-12)  # si tu veux garder la protection

    # Termes masse*enthalpie
    term_mass = m_dot[n] * h_tilde[n] - m_dot_B[n] * h_B[n]

    # Flux pariétal intégré (paroi -> surface)
    term_wall = 2.0 * np.pi * R * H_s * qin_s

    # Interface (interface -> surface liquide)
    term_int = area_surf * kint_liq * (T_int - T_s)

    # Conduction vers le bulk
    if n > 0:
        dTBdx_n = (T_B[n] - T_B[n-1]) / dx
    else:
        dTBdx_n = (T_B[1] - T_B[0]) / dx

    term_cond_bulk = - area_bulk * lambda_B * dTBdx_n

    # Numérateur : puissance (W)
    num = term_mass + term_wall + term_int + term_cond_bulk

    # Dénominateur : rho * cv * volume
    den = rho_liq * cv_liq * area_surf * H_eff

    # Équation : dT/dt = num/(ρ cv V) - (T_s - T_int) * dH_s/dt / H_s
    return (num / den) - ((T_s - T_int) * dH_s_dt / H_eff)


# On détermine la Température à l'interface en supposant l'équilibre thermique et en appliquant la loi de Clausius

def Tsat_de_P_Clausius(Pv_pa, Pref_pa, Tref_K, dh_vap_J_per_kg, M_kg_per_mol):
    """
    Fermeture analytique: T_sat(P) avec Δh_vap ~ cte.
    - Pv_pa : pression cible (Pa)
    - Pref_pa, Tref_K : point de référence (e.g. (1 atm, T_ebullition))
    - dh_vap_J_per_kg : chaleur latente (J/kg)
    - M_kg_per_mol : masse molaire
    """
    R_u = 8.314462618  # J/mol/K
    R_spec = R_u / M_kg_per_mol
    invT = (1.0/Tref_K) - (R_spec/dh_vap_J_per_kg) * np.log(Pv_pa / Pref_pa)
    return 1.0 / invT


# Dôme vapeur (gaz parfait)

# Equation 35


def dTv_dt(kwv, Aw, Twv, Tv, k_v_int_val, Tint, cv_v, Vv, rho_v, R):
    return (kwv * Aw * (Twv - Tv) - np.pi*R**2 * k_v_int_val * (Tv - Tint)) / clamp_den(Vv * rho_v * cv_v)


# Equation 34

def dP_dt(Tv, m_dot_ph, Vv, Pv, dTv_dt_val, r_spec, rho_l):
    left = Vv / clamp_den(r_spec * Tv)
    termT = Vv * (Pv / clamp_den(r_spec * (Tv**2))) * dTv_dt_val
    rhs = m_dot_ph * (1.0 - Pv / clamp_den(r_spec * Tv * rho_l))
    return (rhs + termT) / left


########  Paroi  #######
# Equation 36
# On adapte l'équation 36 pour les différents murs que l'on considère, étant donné que pour le terme de conduction il y a une dépendance des murs adjacents

def dTw_dt_liq(i, qext_i, qin_i, delta_w, c_w, lambda_w, Tw, rho_w, dx):
    """
    dTw_i/dt côté liquide : (q_ext - q_in)/(δ_w ρ_w c_w) + (λ_w / (ρ_w c_w)) Δ_x Tw
    avec Δ_x Tw ≈ (T[i+1] - 2T[i] + T[i-1]) / dx^2 (schéma 3 points).
    """
    if i == 0:
        lap = (Tw[1] - Tw[0]) / (dx**2)       # bord bas (simple unilatéral)
    elif i == len(Tw) - 1:
        lap = (Tw[i] - Tw[i-1]) / (dx**2)     # bord haut (simple unilatéral)
    else:
        lap = (Tw[i+1] - 2.0 * Tw[i] + Tw[i-1]) / (dx**2)
    return (qext_i - qin_i) / clamp_den(delta_w * rho_w * c_w) + (lambda_w * lap) / clamp_den(rho_w * c_w)


def dTw_dt_vap(qext_vap, qin_vap, delta_w, c_w, lambda_w,
               Tw_vap, Tw_int, rho_w, H_vap):
    """
    dTw_vap/dt (paroi côté vapeur), modèle 1D simple entre Tw_vap et Tw_int.
    """
    return ((qext_vap - qin_vap) / clamp_den(delta_w * rho_w * c_w)
            + lambda_w * (Tw_vap - Tw_int) / clamp_den(H_vap**2 * rho_w * c_w))


def dTw_dt_surface(qext_s, qin_s, delta_w, c_w, lambda_w,
                   Tw_liq, rho_w, Tw_S, Tw_vap, H_S):
    """
    dTw_int/dt (paroi à la surface libre). On référence la cellule i_surface côté liquide.
    """
    return ((qext_s - qin_s) / clamp_den(delta_w * rho_w * c_w)
            + lambda_w * (Tw_S - Tw_liq - (Tw_vap - Tw_S))
            / clamp_den(H_S**2 * rho_w * c_w))


# Débit massique CL – version “def”

# Equation 11


def dm_dot_CL_dt(i, m_dot, rho, g, alpha_v, delta, T_tilde, TB,
                 tau_w_i, C, R, dx):
    """
    d(ṁ_i)/dt dans la couche limite (Eq. 11 de Duan) :
      = 2πR [ ρ g α_v δ_i (T̃_i - TB_i) - τ_w,i ]
        - (C ṁ_i)/(2π R ρ δ_i) * [ 2 ∂ṁ/∂x - (ṁ/δ) ∂δ/∂x ]

    Discrétisation x : amont 1er ordre pour ∂/∂x.
    """
    # Terme source flottabilité - frottement (AVEC le facteur R)
    S = 2.0 * np.pi * R * (rho * g * alpha_v *
                           delta[i] * (T_tilde[i] - TB[i]) - tau_w_i)

    #  Dérivées spatiales discrètes (amont)
    if i == 0:
        dm_dx = (m_dot[1] - m_dot[0]) / dx
        ddelta_dx = (delta[1] - delta[0]) / dx
    else:
        dm_dx = (m_dot[i] - m_dot[i-1]) / dx
        ddelta_dx = (delta[i] - delta[i-1]) / dx

    # ---- Terme convectif (le signe "moins" est déjà dans Eq. 11) ----
    denom = 2.0 * np.pi * R * rho * max(delta[i], 1e-12)
    Conv_pref = (C * m_dot[i]) / denom
    conv_term = 2.0 * dm_dx - (m_dot[i] / max(delta[i], 1e-12)) * ddelta_dx
    return S - Conv_pref * conv_term

############################ RÉSOLUTION DU SYSTÈME ############################

# Ici on prend des constantes de liquides et de vapeur comme constantes étant donné que l'on à pas l'accés à REFPROP le module utilisé par les auteurs
# de l'article pour déterminer leur variation dans le temps


# Paramètres

Nx = 100  # Nombre de divisons sur la hauteur du bulk
num_iterations = 1000  # Nombre de pas de temps que l'on cherche à simuler
dt = 1e-4  # intervalle de temps choisi pour garantir la convergence et limiter le temps de calcul

# Constantes du problèmes

R = 0.2  # m
hauteur = 0.6  # m
delta_w = 0.07  # m
Nx = 30  # Nombre de divisions de la partie fluide
M = 0.0441      # kg/mol pour le propane
r = 8.314 / M   # ≈ 188 J/kg/K
g = 9.81  # m/s^-2


pourcentage_remplissage = 0.85
H_vap = hauteur*(1-pourcentage_remplissage)
# hauteur de chaque couche de liquide
dx = (hauteur*pourcentage_remplissage)/(Nx+1)
Hs = dx
Hcv = dx
qext = 20
gamma = 3.56*10**5
Pref = 1.013*10**5  # Pa
Tref = 291.47  # temperature initiale pour l'instant

# Pour le liquide

nu_liquide = 2.07e-7
rho_liquide = 502.59
lambda_liquide = 0.097
cp_liquide = 2649.5
cv_liquide = 1640.1
alpha_v = 0.003
a_liquide = 7.6e-8
beta_liquide = 1.3*10**(-3)  # A MODIFIER

# Pour la vapeur

nu_vapeur = 4.64e-7
rho_vapeur = 17.291
lambda_vapeur = 0.018
cp_vapeur = 1928.5
cv_vapeur = cp_vapeur - r

# Pour la paroi

rho_w = 7930
lambda_w = 15.24
c_w = 500


# grandeurs dôme/interface/paroi (valeurs de départ)
# On considère que l'on est à l'équilibre thermique avant l'application de flux extérieur

Ti = 291.47  # K
Tw_vap = Ti  # K
Tw_s = Ti   # K
Tv = Ti   # K
Ts = Ti  # K
Tint = Ti   # K
Pv = 800*10**3  # bar
q_ext = 20.0  # W/m^2


# tableaux d'état

temp_bulk = np.full(Nx, Ti)       # TB
temp_moy_cl = np.full(Nx, Ti)       # T_tilde (T̃)
delta = np.zeros(Nx)     # épaisseur CL (évite division par 0)
enthalpie_moy_CL = np.zeros(Nx)        # h̃ (à mettre à jour selon ton modèle)
enthalpie_y = np.zeros(Nx)        # h_y (idem)
enthalpie_B = np.zeros(Nx)        # h_B (idem)
m_point_CL = np.zeros(Nx)          # ṁ_i dans la CL
m_point_y = np.zeros(Nx)          # ṁ_vers_bulk (interface CL→bulk)
m_point_B = np.zeros(Nx)          # ṁ dans le bulk
Tw = np.full(Nx, Ti)       # température mur côté liquide
eta = np.zeros(Nx)          # η = (T̃-TB)/(Tw-TB)

# On crée d'abord des buffers pour stocker les variations temporrelles des différentes grandeurs pour ne pas mettre
# à jour les variables avant que l'état complet à chaque pas de temps ne soit calculé

dtemp_moy_cl = np.zeros(Nx)
ddelta = np.zeros(Nx)
deta = np.zeros(Nx)
dTw = np.zeros(Nx)
dm_CL = np.zeros(Nx)
dTb = np.zeros(Nx)


def unpack_state(y, Nx):
    """
    Transforme le vecteur d'état y (1D) en variables physiques.
    """
    offset = 0

    TB = y[offset:offset+Nx]
    offset += Nx
    Ttilde = y[offset:offset+Nx]
    offset += Nx
    delta = y[offset:offset+Nx]
    offset += Nx
    m_CL = y[offset:offset+Nx]
    offset += Nx
    Tw = y[offset:offset+Nx]
    offset += Nx

    Ts = y[offset]
    offset += 1
    Tv = y[offset]
    offset += 1
    Pv = y[offset]
    offset += 1
    Tw_s = y[offset]
    offset += 1
    Tw_vap = y[offset]
    offset += 1
    Hs = y[offset]
    offset += 1

    return TB, Ttilde, delta, m_CL, Tw, Ts, Tv, Pv, Tw_s, Tw_vap, Hs


def pack_state(TB, Ttilde, delta, m_CL, Tw, Ts, Tv, Pv, Tw_s, Tw_vap, Hs):
    """
    Assemble toutes les variables physiques en un seul vecteur y.
    """
    return np.concatenate([
        TB, Ttilde, delta, m_CL, Tw,
        np.array([Ts, Tv, Pv, Tw_s, Tw_vap, Hs], dtype=float)
    ])


def rhs(t, y):
    """
    RHS du système : dy/dt = f(t, y)
    Implémente toutes les équations de Duan en réutilisant les fonctions déjà définies.
    """
    # --- 1) Dépack de l'état ---
    TB, Ttilde, delta, m_CL, Tw, Ts, Tv, Pv, Tw_s, Tw_vap, Hs = unpack_state(
        y, Nx)

    # On force en 1D np.array pour éviter les surprises
    TB = np.asarray(TB, dtype=float)
    Ttilde = np.asarray(Ttilde, dtype=float)
    delta = np.asarray(delta, dtype=float)
    m_CL = np.asarray(m_CL, dtype=float)
    Tw = np.asarray(Tw, dtype=float)

    # --- 2) Quantités algébriques (pas dans y) ---
    # Enthalpies
    enthalpie_B = cp_liquide * TB
    enthalpie_moy_CL = cp_liquide * Ttilde
    enthalpie_y = cp_liquide * TB  # choix: flux CL->bulk côté bulk
    # --- 2.b) Champ η pour toute la CL ---
    eta = np.zeros_like(Ttilde)
    for i in range(Nx):
        eta[i] = eta_moy(Ttilde[i], TB[i], Tw[i])

    # Débits dans le bulk m_B (Eq. 22) et flux transversal m_y
    m_B = np.zeros_like(m_CL)
    m_y = np.zeros_like(m_CL)

    # m_B: part de la surface vers le bas
    m_B[-1] = m_CL[-1]
    for i in range(Nx-2, -1, -1):
        m_B[i] = m_B[i+1] - (m_CL[i] - m_CL[i-1])

    # m_y: m_y(i) = m_CL(i) - m_CL(i-1)
    for i in range(Nx):
        if i == 0:
            m_y[i] = m_CL[i]
        else:
            m_y[i] = m_CL[i] - m_CL[i-1]

    # Interface : Tint et rho_v
    Tint = Tsat_de_P_Clausius(Pv, Pref, Tref, gamma, M)
    rho_v = Pv / clamp_den(r * Tv)

    # --- 3) Buffers pour les dérivées ---
    dTB = np.zeros_like(TB)
    dTtilde = np.zeros_like(Ttilde)
    ddelta = np.zeros_like(delta)
    dm_CL_dt = np.zeros_like(m_CL)
    dTw_liq = np.zeros_like(Tw)

    # --- 4) Boucle sur les CV liquides (CL + bulk + paroi) ---
    for i in range(Nx):
        x_i = dx/2.0 + i*dx

        # 4.a) Contrainte pariétale tau_w et coefficient C
        Gr_i = Grashof_x(g, beta_liquide, Tw[i], TB[i], x_i, nu_liquide)
        ub_i = Ub(g, beta_liquide, Tw[i], TB[i], nu_liquide)
        if Gr_i < 1e9:
            tau_w_i = tau_w_lam(rho_liquide, ub_i, g, beta_liquide,
                                Tw[i], TB[i], x_i, nu_liquide)
            C_i = 1.37
        else:
            tau_w_i = tau_w_turb(rho_liquide, ub_i, g, beta_liquide,
                                 Tw[i], TB[i], x_i, nu_liquide)
            C_i = 3.25

        # 4.b) Équation (11) : dm_CL/dt
        dm_CL_dt[i] = dm_dot_CL_dt(
            i, m_dot=m_CL, rho=rho_liquide, g=g, alpha_v=alpha_v,
            delta=delta, T_tilde=Ttilde, TB=TB,
            tau_w_i=tau_w_i, C=C_i, R=R, dx=dx
        )

        # 4.c) Équation (25) : dTB/dt (bulk)
        dTB[i] = dTB_dt_bulk(
            i, lambdaB=lambda_liquide, R=R, rho=rho_liquide, cv=cv_liquide,
            TB=TB, delta=delta, dx=dx,
            m_dot_B=m_B, m_dot_y=m_y,
            h_y=enthalpie_y, h_B=enthalpie_B,
            T_top=Ts
        )

        # 4.d) Flux pariétal local q_in (Eq. 16–18)
        Rax_i = Rayleigh_x(g, alpha_v, Tw[i],
                           TB[i], x_i, nu_liquide, a_liquide)
        q_in_i = q_in_wall(Tw[i], TB[i], Rax_i, x_i, lambda_liquide)

        # 4.e) Paroi liquide : Eq. (36)
        dTw_liq[i] = dTw_dt_liq(
            i, qext_i=q_ext, qin_i=q_in_i,
            delta_w=delta_w, c_w=c_w, lambda_w=lambda_w,
            Tw=Tw, rho_w=rho_w, dx=dx
        )

        # 4.f) CL : Eq. (21) + (15)
        # On prend eta de manière ALGÉBRIQUE : η = (T̃ - TB)/(Tw - TB)
        eta_i = eta_moy(Ttilde[i], TB[i], Tw[i])

        # Premier calcul de dTtilde avec dδ/dt "guess" = 0
        dTtilde_guess = dTtilde_dt(
            i, eta=eta,  # on peut fabriquer un petit vecteur
            dTw_dt=dTw_liq[i], dTB_dt=dTB[i],
            Tw=Tw, TB=TB, ddelta_dt=0.0,
            m_dot=m_CL, m_dot_y=m_y,
            h_tilde=enthalpie_moy_CL, h_y=enthalpie_y,
            q_in_i=q_in_i, R=R, Hcv=Hcv,
            rho=rho_liquide, cv=cv_liquide, delta_i=delta[i]
        )

        # dδ/dt par Eq. (15)
        ddelta_i = ddelta_dt_from_energy(
            i,
            dTtilde_dt_i=dTtilde_guess,
            m_dot=m_CL, m_dot_y=m_y,
            h_tilde=enthalpie_moy_CL, h_y=enthalpie_y,
            q_in_i=q_in_i, R=R, Hcv=Hcv,
            rho=rho_liquide, cv=cv_liquide,
            delta_i=delta[i],
            T_tilde_i=Ttilde[i],
            TB_i=TB[i]
        )

        # On recalcule dTtilde avec le vrai dδ/dt (optionnel, mais plus fidèle)
        dTtilde_i = dTtilde_dt(
            i, eta=eta,
            dTw_dt=dTw_liq[i], dTB_dt=dTB[i],
            Tw=Tw, TB=TB, ddelta_dt=ddelta_i,
            m_dot=m_CL, m_dot_y=m_y,
            h_tilde=enthalpie_moy_CL, h_y=enthalpie_y,
            q_in_i=q_in_i, R=R, Hcv=Hcv,
            rho=rho_liquide, cv=cv_liquide, delta_i=delta[i]
        )

        dTtilde[i] = dTtilde_i
        ddelta[i] = ddelta_i

    # --- 5) SURFACE LIQUIDE + INTERFACE + VAPEUR ---

    # 5.a) Flux paroi -> surface (k_liq(Ts) (Tw_s - Ts))
    q_in_s = k_propane_liq(Ts) * (Tw_s - Ts)

    # 5.b) Paroi côté surface : dTw_s/dt
    dTw_s = dTw_dt_surface(
        q_ext, q_in_s, delta_w, c_w, lambda_w,
        Tw[-1], rho_w, Tw_s, Tw_vap, Hs
    )

    # 5.c) Rayleigh et k_int (vapeur + liquide)
    Ra_vint = Rayleigh_v_int(
        g, alpha_v, Tv, Tint, 2*R,
        nu_vapeur, rho_vapeur, cp_vapeur, lambda_vapeur
    )
    Ra_lint = Rayleigh_l_int(
        g, alpha_v, Ts, Tint, 2*R,
        nu_liquide, a_liquide
    )
    k_vint = k_v_int(Ra_vint, 2*R, lambda_vapeur)
    k_lint = k_l_int(Ra_lint, 2*R, lambda_liquide)

    # 5.d) Débit de phase (Eq. 31)
    m_ph = m_dot_phase(R, gamma, Tv, Ts, Tint, k_vint, k_lint)

    # 5.e) Hauteur liquide (Eq. 30)
    dHs = dHs_dt(m_ph, rho_liquide, R)

    # 5.f) Température de surface liquide (Eq. 29)
    dTs = dTs_dt_surface(
        Nx-1,
        m_B, enthalpie_B,
        m_CL, enthalpie_moy_CL,
        R, Hs,
        q_in_s,
        k_lint,
        Tint, Ts,
        delta,
        lambda_liquide,
        TB,
        dx,
        dHs,
        rho_liquide, cv_liquide
    )

    # 5.g) Vapeur : flux sur Tw_vap
    qin_vap = k_propane_vap(Tv) * (Tw_vap - Tv)

    # Paroi côté vapeur
    dTw_vap_dt = dTw_dt_vap(
        q_ext, qin_vap, delta_w, c_w, lambda_w,
        Tw_vap, Tint, rho_w, H_vap
    )

    # Température vapeur (Eq. 35)
    dTv_dt_val = dTv_dt(
        k_propane_vap(Tv), H_vap * R, Tw_vap, Tv,
        k_vint, Tint, cp_vapeur,
        H_vap * R * 1.0, rho_vapeur, R
    )

    # Pression vapeur (Eq. 34)
    dPv_dt_val = dP_dt(
        Tv, m_ph, H_vap * R * 1.0, Pv,
        dTv_dt_val, r, rho_liquide
    )

    # --- 6) Repack des dérivées ---
    dy = pack_state(
        dTB, dTtilde, ddelta, dm_CL_dt, dTw_liq,
        dTs, dTv_dt_val, dPv_dt_val, dTw_s, dTw_vap_dt, dHs
    )

    return dy


# ---------- ÉTAT INITIAL ----------

TB0 = temp_bulk.copy()
Ttilde0 = temp_moy_cl.copy()
delta0 = delta.copy()
m_CL0 = m_point_CL.copy()
Tw0 = Tw.copy()

y0 = pack_state(
    TB0, Ttilde0, delta0, m_CL0, Tw0,
    Ts, Tv, Pv, Tw_s, Tw_vap, Hs
)

# ---------- RÉSOLUTION TEMPORELLE AVEC BDF ----------

t_final = 3600.0   # par ex. 1 heure
t_eval = np.linspace(0.0, t_final, 300)

sol = solve_ivp(
    fun=lambda t, y: rhs(t, y),
    t_span=(0.0, t_final),
    y0=y0,
    method="BDF",
    t_eval=t_eval,
    atol=1e-4,
    rtol=1e-3
)

# Vérification rapide
print("Status solveur:", sol.message)
print("Nb étapes:", sol.nfev)
