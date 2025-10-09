import numpy as np
import matplotlib.pyplot as plt

# Variables

h = 1e-2

rayon = 0.2
hauteur = 0.6
epaisseur = 0.07
Nx = 30

# Pour le liquide

nu_liquide = 2.07e-7
rho_liquide = 502.59
lambda_liquide = 0.097
cp_liquide = 2649.5
cv_liquide = 1640.1
alpha_v = 1 # A MODIFIER

# Pour la vapeur

nu_vapeur = 4.64e-7
rho_vapeur = 17.291
lambda_vapeur = 0.018 
cp_vapeur = 1928.5 
cv_vapeur = 1533.9

# Pour la paroi

rho_paroi = 7930
lambda_vapeur = 15.24 
cv_vapeur = 500 

g = 9.81


# Variables d'intérêt

delta = 0
temp_moy_cl = 0
temp_bulk = 0

# Fonctions à utiliser

# Quantité de mouvement intégrale pour la couche limite

nombre_grashof = lambda beta, temp_paroi, temp_bulk, x, nu : (g*beta*(temp_paroi-temp_bulk)*x**3)/ (nu**2)

paroi_shear_stress_lam = lambda rho, ub,beta,temp_paroi,temp_bulk,x,nu : rho*(ub**2)*0.953*(nombre_grashof(beta,temp_paroi,temp_bulk,x,nu)**(1/12))

paroi_shear_stress_turb = lambda rho, ub,beta,temp_paroi,temp_bulk,L,nu : rho*(ub**2)*0.684*(nombre_grashof(beta,temp_paroi,temp_bulk,L,nu)**(1/11.9))

qt_mvt_couche_lim = lambda m_point,i,rho,delta,temp_moy_cl,temp_bulk, paroi_shear_stress, C, rayon : 2 * np.pi * (rho * g * alpha_v * delta[i] * (temp_moy_cl[i] - temp_bulk[i]) - paroi_shear_stress)  - (C*m_point[i]/2*np.pi*rayon*rho*delta[i])*(2*((m_point[i]-m_point[i-1])/(hauteur/Nx)) - (m_point[i]/delta[i])*((delta[i]-delta[i-1])/(hauteur/Nx)))




# Flux de chaleur :

Rayleigh = lambda alpha_v, temp_paroi, temp_bulk, x, a_liquide : (g*alpha_v*(temp_paroi-temp_bulk)*x**3)/(nu_liquide*a_liquide)

k_pl_petit_Rayleigh = lambda Rax, lambda_liquide, x : 0.59*(Rax**(1/4))*(lambda_liquide/x)

k_pl_petit_Rayleigh = lambda Rax, lambda_liquide, x : 0.11*(Rax**(1/3))*(lambda_liquide/x)

flux_chaleur = lambda k_pl, temp_paroi,temp_bulk : k_pl*(temp_paroi-temp_bulk)


# Temperature moyenne 

eta_moy = lambda temp_moy_cl, temp_bulk, temp_paroi : (temp_moy_cl - temp_bulk)/(temp_paroi - temp_bulk)

eta_moy_d = lambda temp_moy_cl, temp_paroi, temp_bulk, eta_moy, delta, i : 0 if ((delta[i]- delta[i-1]/(hauteur/Nx)) >=0) else (((temp_moy_cl[i] - temp_moy_cl[i-1])/(h) - eta_moy*((temp_paroi[i]-temp_paroi[i-1])/h) + (1-eta_moy)*(temp_bulk[i]-temp_bulk[i-1])/h)/(temp_paroi[i]-temp_bulk[i]))

dtemp_moy = lambda eta, rho, cv, temp_paroi, temp_bulk, i, m_point, m_point_y, enthalpie_moy, enthalpie_y, flux_entrant, delta : eta[i] * ((temp_paroi[i]-temp_paroi[i-1])/h) + (1-eta_moy)*((temp_bulk[i]-temp_bulk[i-1])/h) + (temp_paroi[i] - temp_bulk[i]) * ((eta[i]-eta[i-1])/h) if (((delta[i]-delta[i-1])/h) >= 0) else (enthalpie_moy[i]*m_point[i] - enthalpie_moy[i-1]*m_point[i-1] - m_point_y[i]* enthalpie_y[i] + 2*np.pi * rayon * (hauteur/Nx)*flux_entrant[i])/(2*np.pi*rayon*(hauteur/Nx)*rho*cv*delta[i])

