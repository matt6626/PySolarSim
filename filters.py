def rc_filter(vin_array, r, c, vout0, dt):
    vout_array = [vout0]
    for vin in vin_array:
        # Ic = C * (Vc - Vc0) / dt
        # Ic = (Vin - Vc) / R
        # (Vin - Vc) / R = C * (Vc - Vc0) / dt
        # Vin - Vc = R * C * (Vc - Vc0) / dt
        # Vin = R * C * (Vc - Vc0) / dt + Vc
        # Vin + R * C * Vc0 / dt = R * C * Vc / dt + Vc
        # Vin + R * C * Vc0 / dt = Vc (R * C / dt + 1)
        # Vc = (Vin + R * C * Vc0 / dt) / (R * C / dt + 1)
        vout_prev = vout_array[-1]
        out = (vin + r * c * vout_prev / dt) / (r * c / dt + 1)
        vout_array.append(out)
    vout_array.pop()
    return vout_array
