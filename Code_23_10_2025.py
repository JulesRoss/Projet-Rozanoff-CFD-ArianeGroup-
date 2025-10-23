import numpy as np
import matplotlib.pyplot as plt

# Variables

R= 0.2 ## m
hauteur = 0.6 ## m
delta_w = 0.07 ## m
Nx = 30  ## Nombre de divisions de la partie fluide
M=18 ## g/mol
r=8.314/M ## Constante spécifique des gaz parfaits

dt = 1e-2  ## intervalle de temps choisi pour garantir la convergence et limiter le temps de calcul
dx=hauteur/Nx  ## hauteur de chaque couche de liquide

# Pour le liquide

nu_liquide = 2.07e-7
rho_liquide = 502.59
lambda_liquide = 0.097
cp_liquide = 2649.5
cv_liquide = 1640.1
alpha_v = 1 # A MODIFIER
a_liquide = 1 # A MODIFIER
beta_liquide = 1 # A MODIFIER

# Pour la vapeur

nu_vapeur = 4.64e-7
rho_vapeur = 17.291
lambda_vapeur = 0.018 
cp_vapeur = 1928.5 
cv_vapeur = 1533.9

# Pour la paroi

rho_w = 7930
lambda_w = 15.24 
c_w = 500 


g = 9.81


# Variables d'intérêt

delta = 0
temp_moy_cl = 0
temp_bulk = 0
Tint= 273.15 ### Je suppose que c'est la température d'ébulition à la pression actuelle de la vapeur
qext= 20 ## W/m**2
num_iterations= 1000



##################### FONCTIONS #################################

#################### COUCHE LIMITE ##############################

# Quantité de mouvement intégrale pour la couche limite

Ub = lambda g, beta, temp_paroi, temp_bulk, nu : (g*beta*(temp_paroi-temp_bulk)*nu)**(1/3)

nombre_grashof = lambda beta, temp_paroi, temp_bulk, x, nu : (g*beta*(temp_paroi-temp_bulk)*x**3)/ (nu**2) ##   Tsuji Nagano

paroi_shear_stress_lam = lambda rho, ub,beta,temp_paroi,temp_bulk,x,nu : rho*(ub**2)*0.953*(nombre_grashof(beta,temp_paroi,temp_bulk,x,nu)**(1/12)) ## Tsuji Nagano pour Gr< 10**9

paroi_shear_stress_turb = lambda rho, ub,beta,temp_paroi,temp_bulk,L,nu : rho*(ub**2)*0.684*(nombre_grashof(beta,temp_paroi,temp_bulk,L,nu)**(1/11.9))  ## Tsuji Nagano pour Gr>10**9

d_debit_massique = lambda m_point,i,rho,delta,temp_moy_cl,temp_bulk, paroi_shear_stress, C, R : 2 * np.pi * (rho * g * alpha_v * delta[i] * (temp_moy_cl[i] - temp_bulk[i]) - paroi_shear_stress)  - (C*m_point[i]/2*np.pi*R*rho*delta[i])*(2*((m_point[i]-m_point[i-1])/dx) - (m_point[i]/delta[i])*((delta[i]-delta[i-1])/dx))  ##  11




# Flux de chaleur :

Rayleigh = lambda alpha_v, temp_paroi, temp_bulk, x, a_liquide : (g*alpha_v*(temp_paroi-temp_bulk)*x**3)/(nu_liquide*a_liquide)  ## 18

k_pl_10_5_9 = lambda Rax, lambda_liquide, x : 0.59*(Rax**(1/4))*(lambda_liquide/x)  ## 17

k_pl_10_9_13 = lambda Rax, lambda_liquide, x : 0.11*(Rax**(1/3))*(lambda_liquide/x)  ##  17

q_in = lambda temp_paroi,temp_bulk ,Rax,x  : k_pl_10_5_9(Rax, lambda_liquide, x)*(temp_paroi-temp_bulk) if 10**9>Rax or Rax>10**5 else k_pl_10_9_13(Rax, lambda_liquide, x)*(temp_paroi-temp_bulk) ## 16


# Temperature moyenne 

eta_moy = lambda temp_moy_cl, temp_bulk, temp_paroi : (temp_moy_cl - temp_bulk)/(temp_paroi - temp_bulk) ## 19

d_delta=5

d_eta_moy = lambda d_temp_paroi,d_temp_bulk, temp_paroi, temp_bulk, eta_moy,d_delta, i ,d_temp_moy : 0 if d_delta>=0 else ((d_temp_moy - eta_moy*d_temp_paroi + (1-eta_moy)*d_temp_paroi)/(temp_paroi[i]-temp_bulk[i]))  ## 20   ############ CORRIGER VARIATION PAR RAPPORT AU TEMPS ET PAS A L'ESPACE

d_temp_moy = lambda eta, rho, cv, d_temp_paroi , d_temp_bulk , temp_paroi, temp_bulk, i, m_point, m_point_y, enthalpie_moy, enthalpie_y, flux_entrant, d_delta, delta,R : (eta[i] *d_temp_paroi + (1-eta[i])*d_temp_bulk + (temp_paroi[i] - temp_bulk[i])) * ((eta[i]-eta[i-1])/dt) if d_delta>=0 else (enthalpie_moy[i]*m_point[i] - enthalpie_moy[i-1]*m_point[i-1] - m_point_y[i]* enthalpie_y[i] + 2*np.pi * R * dx*flux_entrant[i])/(2*np.pi*R*dx*rho*cv*delta[i])   ##21   ############ CORRIGER VARIATION PAR RAPPORT AU TEMPS ET PAS A L'ESPACE

## On suppose que la couhe limite se développe (delta[i]-delta[i-1]>0)dans un premier temps, puis on résoud les équations, on réinjecte les résultats pour calculer le débit massique puis on avise

d_delta_i = lambda d_temp_moy,  m_point, m_point_y, enthalpie_moy, enthalpie_y, flux_entrant, R, rho, delta, i , temp_bulk, temp_moy_cl ,cv : ((m_point[i]*enthalpie_moy[i]-m_point[i-1]*enthalpie_moy[i-1]-m_point_y[i]*enthalpie_y[i])/(2*np.pi*R*dx*rho*cv_liquide)- flux_entrant/(rho*cv_liquide)-delta[i]*d_temp_moy)/(temp_moy_cl[i]-temp_bulk[i])## 15



############################# BULK ############################


bilan_masse = lambda mB,m_point_y, i : mB[i]-m_point_y ## 22

#equation de conservation d'énergie

d_temp_moy_Bulk = lambda lambdaB,R,eta, rho, cv, temp_bulk, i, m_point_B, m_point_y, enthalpie_y, enthalpie_B, flux_entrant, delta : (m_point_B[i]*enthalpie_B[i]-m_point_B[i-1]*enthalpie_B[i-1]-m_point_y[i]*enthalpie_y[i]+np.pi*(R-delta)**2*lambdaB*((temp_bulk[i]-temp_bulk[i-1])/dx-(temp_bulk[i-1]-temp_bulk[i-2])/dx))/(np.pi*rho*cv*dx*(R-delta[i])**2)  ## 25




############################### SURFACE #######################

Rayleigh_v_int = lambda alpha_v, temp_vap, temp_int, D, nu_vapeur, a_vapeur : (g*alpha_v*(temp_vap-temp_int)*D**3)/(nu_vapeur*a_vapeur)  ## 32

Rayleigh_l_int = lambda alpha_l, temp_S, temp_int, D, nu_liquide, a_liquide : (g*alpha_l*(temp_int-temp_S)*D**3)/(nu_liquide*a_liquide)  ## 32 

kv_int = lambda Ra_v_int, D,lambda_vapeur: 0.54*Ra_v_int**(1/4)*lambda_vapeur/D  ## 32

kint_liq = lambda Ra_l_int, D,lambda_liq: 0.27*Ra_l_int**(1/4)*lambda_liq/D  ## 32

dm_phase= lambda R, gamma, temp_vap, temp_S, temp_int : np.pi*R**2*(kv_int*(temp_vap-temp_int)-kint_liq*(temp_int-temp_S))/gamma

d_H_S =  lambda  dm_ph, rho, R : dm_ph/(rho*np.pi*R**2)

d_Ts_dt = lambda  m_point_B, enthalpie, m_point, enthalpie_B, n, R,H_S,qin_s,temp_int, temp_S, R_delta, lambda_B, temp_B,            : (m_point_B[n]*enthalpie_B[n]- m_point[n]*enthalpie[n]+2*np.pi*R*H_S*qin_s+np.pi*R**2*kint_liq*(temp_int-temp_S)- np.pi*(R_delta[i])**2*lambda_B*(temp_B[n]-temp_B[n-1])/(h/Nx))/(np.pi*R**2*H_S)-(temp_S-temp_int)*d_H_S/H_S



########################## VAPOR GAS REPONSE ############################



dTv = lambda kwv,Aw,Twv,Tv,kvint,Tint,cv,Vv : (kwv*Aw*(Twv-Tv) - np.pi()(R**2)*kvint(Tv-Tint))/(Vv*rho_vapeur*cv_liquide) ## 35

dP = lambda Tv, dm_ph, Vv, Pv, dTv : r*Tv*((dm_ph/Vv)(1-((Pv/(r*Tv))//rho_liquide)) - (-Pv/r(Tv**2))*dTv) ## 34



######################### WALL #################################


d_temp_mur = lambda qext, qin, delta_w, cw, lambdaw, T_wall , i, rho_w: (qext-qin[i])/(delta_w*rho_w*cw) + lambdaw*(T_wall[i]-2*T_wall[i-1]-T_wall[i-2])/(dx**2*rho_w*cw) ### COUCHE LIMITE 

d_temp_mur_vap = lambda qext_vap, qin_vap, delta_w, cw, lambdaw, T_wall_vap, T_wall_int , i, rho_w,hauteur_vap: (qext_vap-qin_vap)/(delta_w*rho_w*cw) + lambdaw*(T_wall_vap-T_wall_int)/((hauteur_vap)**2*rho_w*cw) ### VAPEUR

d_temp_mur_int = lambda qext_int, qin_int, delta_w, cw, lambdaw, T_wall , n, rho_w, T_wall_int, T_wall_vap, hauteur_interface: (qext_int-qin_int)/(delta_w*rho_w*cw) + lambdaw*(T_wall_int-T_wall[n] -(T_wall_vap-T_wall_int))/((hauteur_interface)**2*rho_w*cw) ### INTERFACE




############################ RESOLUTION DU SYSTEME ###########################################

temp_bulk=np.array(Nx)
temp_moy_cl=np.array(Nx)
delta=np.array(Nx)
enthalpie_moy_CL=np.array(Nx)
enthalpie_y= np.array(Nx)
enthalpie_B=np.array(Nx)
m_point_CL=np.array(Nx)
m_point_y=np.array(Nx)
m_point_B=np.array(Nx)
Tw=np.array(Nx)
eta=np.array(Nx)

Tw_vap=2   ##K
Tw_int=2  ##K
Tv=2 ##K
Pv=2   ## bar
q_ext= 20 ## W/m**2

### Process à itérer sur le nombre de time steps souhaités
### On resoud de en bas à gauche du réservoir à en haut à gauche en calculant la varition de température/taille de la couche limite et la température du mur
### Puis on résout les équations à la surface du liquide et pour la phase vapeur. 
### Enfin on résout les équations de flux dans toutes les différents couche du liqide au repos en commançant par la couche N, tout en haut et en descendant vers la couche 0*


### Phase d'initalisation
### On calcule la taille de couche limite et la température de la couche tout en bas du réservoir


for iterations in range(num_iterations):
    #### INITIALISATION ####
    ## Etape essentielle car les équations utilisent souvent variable[i-1] en paramètre d'entrée et donc il faut les ajuster pour les cas limites

    Rax_bas_gauche = Rayleigh(alpha_v,Tw[0],temp_bulk[0],dx/2,a_liquide)
    q_in_bas_gauche = q_in(Tw[0],temp_bulk[0],Rax_bas_gauche,dx/2)

    ## Variation de Tw[0]

    d_temp_mur_bas_gauche = lambda qext, qin, delta_w, cw, lambdaw, T_wall , rho_w: (qext-qin[0])/(delta_w*rho_w*cw) + lambdaw*(T_wall[1]-T_wall[0])/(dx**2*rho_w*cw) ### COUCHE LIMITE 
    Rax= Rayleigh(alpha_v,Tw[0],temp_bulk[0],dx/2,a_liquide)
    qin = q_in(Tw[0],temp_bulk[0],Rax,dx/2)
    Tw[0] = Tw[0] + dt * d_temp_mur_bas_gauche(qext,qin,delta_w,c_w, lambda_w,Tw[0],rho_w)

    ## Variation du débit massique dans la couche limite
    
    Gr_bas_gauche = nombre_grashof(beta_liquide,Tw[0],temp_bulk[0],dx/2,nu_liquide)
    Ub = Ub(g,beta_liquide,Tw[0],temp_bulk[0],nu_liquide)
    paroi_shear_stress_bas_gauche = paroi_shear_stress_lam(rho_liquide,Ub,beta_liquide,Tw[0],temp_bulk[0],dx/2,nu_liquide) if Gr_bas_gauche<10**9 else paroi_shear_stress_turb(rho_liquide,Ub,beta_liquide,Tw[0],temp_bulk[0],dx/2,nu_liquide)
    C = 1.37 if Gr_bas_gauche<10**9 else 3.25
    d_debit_massique_bas_gauche = lambda m_point,rho,delta,temp_moy_cl,temp_bulk, paroi_shear_stress, C, R : 2 * np.pi * (rho * g * alpha_v * delta[0] * (temp_moy_cl[0] - temp_bulk[0]) - paroi_shear_stress)  - (C*m_point[0]/2*np.pi*R*rho*delta[0])*(m_point[0]/dx) if delta[0]!=0 else 2 * np.pi * (rho * g * alpha_v * delta[0] * (temp_moy_cl[0] - temp_bulk[0]) - paroi_shear_stress)  ## Si l'épaisseur de la couche limite est nulle alors le dénit massique qui la traverse l'est aussi et donc le second terme est nul.
    m_point_CL[0] = m_point_CL[0] + dt*d_debit_massique_bas_gauche(m_point_CL[0],rho_liquide,delta,temp_moy_cl[0],temp_bulk[0],paroi_shear_stress_bas_gauche,C,R)

    ## Variation de la taille de la couche limite avec calcul de la variation de la température au sein de la couche limite
    ## On suppose initalement que ddelta/dt >= 0 :

    eta_moy_bas_gauche = eta_moy(temp_moy_cl[0],temp_bulk[0],Tw[0])

    d_delta=1 ## n'importe quelle valeur plus grande que 0 convient ici

    d_eta_moy_bas_gauche = d_eta_moy(temp_moy_cl[0],Tw[0],temp_bulk[0],eta_moy_bas_gauche,d_delta,0,0)
    d_temp_moy_bas_gauche = d_temp_moy(eta_moy_bas_gauche,rho_liquide,cv_liquide,Tw[0],temp_bulk[0],0,m_point_CL,m_point_y,enthalpie_moy_CL,enthalpie_y,q_in,d_delta,R)
    ## On suppose que la couhe limite se développe (delta[i]-delta[i-1]>0)dans un premier temps, puis on résoud les équations, on réinjecte les résultats pour calculer le débit massique puis on avise

    d_delta_bas_gauche =  d_delta_i(d_temp_moy_bas_gauche,m_point_CL,m_point_y,enthalpie_moy_CL,enthalpie_y,q_in_bas_gauche,R,rho_liquide,delta[0],0,temp_bulk[0],temp_moy_cl[0],cv_liquide)
    

    if d_delta_bas_gauche < 0 : 
        d_delta= -1
        d_temp_moy_bas_gauche = d_temp_moy(eta_moy_bas_gauche,rho_liquide,cv_liquide,Tw[0],temp_bulk[0],0,m_point_CL,m_point_y,enthalpie_moy_CL,enthalpie_y,q_in,d_delta,R)
        d_eta_moy_bas_gauche = d_eta_moy(temp_moy_cl[0],Tw[0],temp_bulk[0],eta_moy_bas_gauche,d_delta,0,d_temp_moy_bas_gauche)
        d_delta_bas_gauche =  d_delta_i(d_temp_moy_bas_gauche,m_point_CL,m_point_y,enthalpie_moy_CL,enthalpie_y,q_in_bas_gauche,R,rho_liquide,delta[0],0,temp_bulk[0],temp_moy_cl[0],cv_liquide)
    
    temp_moy_cl[0]= temp_moy_cl[0] + d_temp_moy_bas_gauche*dt
    delta[0]= delta[0] + d_delta_bas_gauche*dt

    for i in range(1,Nx):
        Rax_bas_gauche = Rayleigh(alpha_v,Tw[i],temp_bulk[i],dx/2+i*dx,a_liquide)
        q_in_i = q_in(Tw[i],temp_bulk[i],Rax_bas_gauche,dx/2+i*dx)

        ## Variation de Tw[i] 
        Rax= Rayleigh(alpha_v,Tw[i],temp_bulk[i],dx/2+i*dx,a_liquide)
        qin = q_in(Tw[i],temp_bulk[i],Rax,dx/2+i*dx)
        Tw[i] = Tw[i] + dt * d_temp_mur(qext,qin,delta_w,c_w, lambda_w,Tw,i,rho_w)

        ## Variation du débit massique dans la couche limite
    
        Gr_i = nombre_grashof(beta_liquide,Tw[i],temp_bulk[i],dx/2+i*dx,nu_liquide)
        Ub = Ub(g,beta_liquide,Tw[i],temp_bulk[i],nu_liquide)
        paroi_shear_stress_i = paroi_shear_stress_lam(rho_liquide,Ub,beta_liquide,Tw[i],temp_bulk[i],dx/2+i*dx,nu_liquide) if Gr_bas_gauche<10**9 else paroi_shear_stress_turb(rho_liquide,Ub,beta_liquide,Tw[i],temp_bulk[i],dx/2+i*dx,nu_liquide)
        C = 1.37 if Gr_bas_gauche<10**9 else 3.25
        m_point_CL[i] = m_point_CL[i] + dt*d_debit_massique(m_point_CL,i,rho_liquide,delta,temp_moy_cl,temp_bulk,paroi_shear_stress_i,C,R)

        ## Variation de la taille de la couche limite avec calcul de la variation de la température au seind e la couche limite
        ## On suppose initalement que ddelta/dt >= 0 :

        eta_moy_i = eta_moy(temp_moy_cl[i],temp_bulk[i],Tw[i])

        d_delta=1 ## n'importe quelle valeur plus grande que 0 convient icic

        d_eta_moy_bas_gauche = d_eta_moy(temp_moy_cl[i],Tw[i],temp_bulk[i],eta_moy_bas_gauche,d_delta,i,0)
        d_temp_moy_bas_gauche = d_temp_moy(eta_moy_bas_gauche,rho_liquide,cv_liquide,Tw[i],temp_bulk[i],i,m_point_CL,m_point_y,enthalpie_moy_CL,enthalpie_y,q_in,d_delta,R)
        ## On suppose que la couhe limite se développe (delta[i]-delta[i-1]>0)dans un premier temps, puis on résoud les équations, on réinjecte les résultats pour calculer le débit massique puis on avise

        d_delta_bas_gauche =  d_delta_i(d_temp_moy_bas_gauche,m_point_CL,m_point_y,enthalpie_moy_CL,enthalpie_y,q_in_bas_gauche,R,rho_liquide,delta,i,temp_bulk,temp_moy_cl,cv_liquide)
    

        if d_delta_bas_gauche < 0 : 
          d_delta= -1
          d_temp_moy_bas_gauche = d_temp_moy(eta_moy_bas_gauche,rho_liquide,cv_liquide,Tw,temp_bulk,i,m_point_CL,m_point_y,enthalpie_moy_CL,enthalpie_y,q_in,d_delta,R)
          d_eta_moy_bas_gauche = d_eta_moy(temp_moy_cl,Tw,temp_bulk,eta_moy_bas_gauche,d_delta,i,d_temp_moy_bas_gauche)
          d_delta_bas_gauche =  d_delta_i(d_temp_moy_bas_gauche,m_point_CL,m_point_y,enthalpie_moy_CL,enthalpie_y,q_in_bas_gauche,R,rho_liquide,delta,i,temp_bulk,temp_moy_cl,cv_liquide)
    
        temp_moy_cl[i]= temp_moy_cl[0] + d_temp_moy_bas_gauche*dt
        delta[i]= delta[i] + d_delta_bas_gauche*dt

    #### INTERFACE FLUIDE VAPEUR AVEC CALCUL DES GRANDES POUR L'INTERFACE ET LA PHASE VAPEUR


    




















