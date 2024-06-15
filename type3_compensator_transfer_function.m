pkg load symbolic

syms s
syms adc fp r1 r2 r3 c1 c2 c3 wp real positive

denom = 0 == s^3*(r1 + r3 + adc*wp*r1*r3*r3) + s^2*(((r1+r3)*c2+c1*c2*r2*(r1+r3)*wp+c1*r2+adc*wp*c2*r1*r3*(c1+c3)+adc*wp*c1*c3*r1*r2)/(c1*c2*r2)) + s*((c2*(r1+r3)*wp+1+wp*c1*r2+(adc*wp*r1)*(c1+c3))/(c1*c2*r2)) + wp/(c1*c2*r2)

roots = solve(denom, s)


#H_OL = A_dc / (1 + s / (wp))
#B = (r1 * r3 * c1 * s * (s + (c1 + c2)/(c1*c2*r2)) * (s + 1/(r3*c3))) / ((r1+r3)*(s+1/(r2*c2))*(s+1/((r1+r3)*c3)))

#H_compensator = H_OL / (1 + H_OL * B)

#factor(H_compensator, s)
#[num, den] = numden(H_compensator)

#sol = solve([eqn1, eqn2, eqn3], [v_c1, v_c2, v_c3]);
#v_c1 = matlabFunction(factor(sol.v_c1))
#v_c2 = matlabFunction(factor(sol.v_c2))
#v_c3 = matlabFunction(factor(sol.v_c3))

#v_c1 = matlabFunction(factor(sol.v_c1))
#v_c2 = matlabFunction(factor(sol.v_c2))
#v_c3 = matlabFunction(factor(sol.v_c3))
