    def k_propane_liq(T):
        """Conductivité thermique [W/m/K] du propane liquide (230–320 K)."""
        return max(0.146 - 1.8e-4 * T, 0.05)

    def k_propane_vap(T):
        """Conductivité thermique [W/m/K] du propane vapeur (250–350 K)."""
        return 7.2e-3 + 3.5e-5 * T
    
    def qin_vap(Tv, Tw_vap):
        """q_in_vap = k_vap * (Tv - Tw_vap) """
        k_vap = k_propane_vap(Tv)
        
        return k_vap * (Tv - Tw_vap) 
    kwv
    Aw
    cv_v
    Vv
    rho_v
    r_spec
    rho_1c
    c_w
    delta_w
    lambda_w


    for i in range(1, Nx):

        qin_vap_i = qin_vap(Tv, Tw_vap)

        dTwv_dt = dTw_dt_vap(q_ext, qin_vap_i, delta_w, c_w, lambda_w, Tw_vap, Tw_int, rho_w, H_vap)
        Tw_vap += dTwv_dt * dt

        dTw_int_dt = dTw_dt_int(i, q_ext, qin_int, delta_w, c_w, lambda_w, Tw_liq, rho_w, Tw_int, Tw_vap, H_int)
        Tw_int += dTw_int_dt * dt
                          
        dTv_dt_i = dTv_dt(kwv, Aw, Tw_vap, Tv, k_v_int_val, Tw_int, cv_v, Vv, rho_v)
        Tv += dTv_dt_i * dt
        
        dP_dt_i = dP_dt(Tv, m_dot_ph, Vv, Pv, dTv_dt_i, r_spec, rho_l)
        Pv += dP_dt_i * dt

        rho_v = Pv / (r_spec * Tv)