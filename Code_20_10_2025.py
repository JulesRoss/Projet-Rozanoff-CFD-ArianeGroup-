import numpy as np
import matplotlib.pyplot as plt

# Variables

h = 1e-2

rayon = 0.2
hauteur = 0.6
epaisseur = 0.07
Nx = 30
M=18 ## g/mol
r=8.314/M

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
lambda_paroi = 15.24 
cv_paroi = 500 

g = 9.81


# Variables d'intérêt

delta = 0
temp_moy_cl = 0
temp_bulk = 0
Tint= 273.15 ### Je suppose que c'est la température d'ébulition à la pression actuelle de la vapeur

##################### FONCTIONS #################################

#################### COUCHE LIMITE ##############################

# Quantité de mouvement intégrale pour la couche limite

nombre_grashof = lambda beta, temp_paroi, temp_bulk, x, nu : (g*beta*(temp_paroi-temp_bulk)*x**3)/ (nu**2) ##   Tsuji Nagano

paroi_shear_stress_lam = lambda rho, ub,beta,temp_paroi,temp_bulk,x,nu : rho*(ub**2)*0.953*(nombre_grashof(beta,temp_paroi,temp_bulk,x,nu)**(1/12)) ## Tsuji Nagano

paroi_shear_stress_turb = lambda rho, ub,beta,temp_paroi,temp_bulk,L,nu : rho*(ub**2)*0.684*(nombre_grashof(beta,temp_paroi,temp_bulk,L,nu)**(1/11.9))  ## Tsuji Nagano

qt_mvt_couche_lim = lambda m_point,i,rho,delta,temp_moy_cl,temp_bulk, paroi_shear_stress, C, rayon : 2 * np.pi * (rho * g * alpha_v * delta[i] * (temp_moy_cl[i] - temp_bulk[i]) - paroi_shear_stress)  - (C*m_point[i]/2*np.pi*rayon*rho*delta[i])*(2*((m_point[i]-m_point[i-1])/(hauteur/Nx)) - (m_point[i]/delta[i])*((delta[i]-delta[i-1])/(hauteur/Nx)))  ##  11




# Flux de chaleur :

Rayleigh = lambda alpha_v, temp_paroi, temp_bulk, x, a_liquide : (g*alpha_v*(temp_paroi-temp_bulk)*x**3)/(nu_liquide*a_liquide)  ## 18

k_pl_10_5_9 = lambda Rax, lambda_liquide, x : 0.59*(Rax**(1/4))*(lambda_liquide/x)  ## 17

k_pl_10_9_13 = lambda Rax, lambda_liquide, x : 0.11*(Rax**(1/3))*(lambda_liquide/x)  ##  17

flux_chaleur = lambda k_pl_10_5_9, k_pl_10_9_13, temp_paroi,temp_bulk ,Rax  : k_pl_10_5_9*(temp_paroi-temp_bulk) if 10**9>Rax or Rax>10**5 else k_pl_10_9_13*(temp_paroi-temp_bulk) ## 16


# Temperature moyenne 

eta_moy = lambda temp_moy_cl, temp_bulk, temp_paroi : (temp_moy_cl - temp_bulk)/(temp_paroi - temp_bulk) ## 19

d_delta=5

d_eta_moy = lambda temp_moy_cl, temp_paroi, temp_bulk, eta_moy, d_delta, i : 0 if d_delta>=0 else (((temp_moy_cl[i] - temp_moy_cl[i-1])/(h) - eta_moy*((temp_paroi[i]-temp_paroi[i-1])/h) + (1-eta_moy)*(temp_bulk[i]-temp_bulk[i-1])/h)/(temp_paroi[i]-temp_bulk[i]))  ## 20

d_temp_moy = lambda eta, rho, cv, temp_paroi, temp_bulk, i, m_point, m_point_y, enthalpie_moy, enthalpie_y, flux_entrant, d_delta, delta : eta[i] * ((temp_paroi[i]-temp_paroi[i-1])/h) + (1-eta[i])*((temp_bulk[i]-temp_bulk[i-1])/h) + (temp_paroi[i] - temp_bulk[i]) * ((eta[i]-eta[i-1])/h) if d_delta>=0 else (enthalpie_moy[i]*m_point[i] - enthalpie_moy[i-1]*m_point[i-1] - m_point_y[i]* enthalpie_y[i] + 2*np.pi * rayon * (hauteur/Nx)*flux_entrant[i])/(2*np.pi*rayon*(hauteur/Nx)*rho*cv*delta[i])   ##21

## On suppose que la couhe limite se développe (delta[i]-delta[i-1]>0)dans un premier temps, puis on résoud les équations, on réinjecte les résultats pour calculer le débit massique puis on avise

d_delta_i = lambda d_temp_moy,  m_point, m_point_y, enthalpie_moy, enthalpie_y, flux_entrant, R, rho, delta, i , temp_bulk, temp_moy_cl ,cv : ((m_point[i]*enthalpie_moy[i]-m_point[i-1]*enthalpie_moy[i-1]-m_point_y[i]*enthalpie_y[i])/(2*np.pi*R*hauteur/Nx*rho*cv_liquide)- flux_entrant/(rho*cv_liquide)-delta[i]*d_temp_moy)/(temp_moy_cl[i]-temp_bulk[i])## 15



############################# BULK ############################


bilan_masse = lambda mB,m_point_y, i : mB[i]-m_point_y ## 22

#equation de conservation d'énergie

d_temp_moy_Bulk = lambda lambdaB,R,eta, rho, cv, temp_bulk, i, m_point_B, m_point_y, enthalpie_y, enthalpie_B, flux_entrant, delta : (m_point_B[i]*enthalpie_B[i]-m_point_B[i-1]*enthalpie_B[i-1]-m_point_y[i]*enthalpie_y[i]+np.pi*(R-delta)**2*lambdaB*((temp_bulk[i]-temp_bulk[i-1])/(hauteur/Nx)-(temp_bulk[i-1]-temp_bulk[i-2])/(hauteur/Nx)))/(np.pi*rho*cv*(hauteur/Nx)*(R-delta[i])**2)  ## 25




############################### SURFACE #######################

Rayleigh_v_int = lambda alpha_v, temp_vap, temp_int, D, nu_vapeur, a_vapeur : (g*alpha_v*(temp_vap-temp_int)*D**3)/(nu_vapeur*a_vapeur)  ## 32

Rayleigh_l_int = lambda alpha_l, temp_S, temp_int, D, nu_liquide, a_liquide : (g*alpha_l*(temp_int-temp_S)*D**3)/(nu_liquide*a_liquide)  ## 32 

kv_int = lambda Ra_v_int, D,lambda_vapeur: 0.54*Ra_v_int**(1/4)*lambda_vapeur/D  ## 32

kint_liq = lambda Ra_l_int, D,lambda_liq: 0.27*Ra_l_int**(1/4)*lambda_liq/D  ## 32

dm_phase= lambda R, gamma, temp_vap, temp_S, temp_int : np.pi*R**2*(kv_int*(temp_vap-temp_int)-kint_liq*(temp_int-temp_S))/gamma

d_H_S =  lambda  dm_ph, rho, R : dm_ph/(rho*np.pi*R**2)

d_Ts_dt = lambda  m_point_B, enthalpie, m_point, enthalpie_B, n, R,H_S,qin_s,temp_int, temp_S, R_delta, lambda_B, temp_B,            : (m_point_B[n]*enthalpie_B[n]- m_point[n]*enthalpie[n]+2*np.pi*R*H_S*qin_s+np.pi*R**2*kint_liq*(temp_int-temp_S)- np.pi*(R_delta[i])**2*lambda_B*(temp_B[n]-temp_B[n-1])/(h/Nx))/(np.pi*R**2*H_S)-(temp_S-temp_int)*d_H_S/H_S



########################## VAPOR GAS REPONSE ############################



dTv = lambda kwv,Aw,Twv,Tv,kvint,Tint,cv,Vv : (kwv*Aw*(Twv-Tv) - np.pi()(rayon**2)*kvint(Tv-Tint))/(Vv*rho_vapeur*cv_liquide) ## 35

dP = lambda Tv, dm_ph, Vv, Pv, dTv : r*Tv*((dm_ph/Vv)(1-((Pv/(r*Tv))//rho_liquide)) - (-Pv/r(Tv**2))*dTv) ## 34



######################### WALL #################################


d_temp_mur = lambda qext, qin, delta_w, cw, lambdaw, T_wall , i, rho_w: (qext[i]-qin[i])/(delta_w*rho_w*cw) + lambdaw*(T_wall[i]-2*T_wall[i-1]-T_wall[i-2])/((hauteur/Nx)**2*rho_w*cw) ### COUCHE LIMITE 

d_temp_mur_vap = lambda qext_vap, qin_vap, delta_w, cw, lambdaw, T_wall_vap, T_wall_int , i, rho_w,hauteur_vap: (qext_vap-qin_vap)/(delta_w*rho_w*cw) + lambdaw*(T_wall_vap-T_wall_int)/((hauteur_vap)**2*rho_w*cw) ### VAPEUR

d_temp_mur_int = lambda qext_int, qin_int, delta_w, cw, lambdaw, T_wall , n, rho_w, T_wall_int, T_wall_vap, hauteur_interface: (qext_int-qin_int)/(delta_w*rho_w*cw) + lambdaw*(T_wall_int-T_wall[n] -(T_wall_vap-T_wall_int))/((hauteur_interface)**2*rho_w*cw) ### INTERFACE




############################ RESOLUTION DU SYSTEME ###########################################

temp_bulk=np.array(Nx)
temp_moy_cl=np.array(Nx)
delta=np.array(Nx)
enthalpie_moy=np.array(Nx)
enthalpie_y= np.array(Nx)
enthalpie_B=np.array(Nx)
m_point_CL=np.array(Nx)
m_point_B=np.array(Nx)
Tw=np.array(Nx)

Tw_vap=2   ##K
Tw_int=2  ##K
Tv=2 ##K
Pv=2   ## bar
