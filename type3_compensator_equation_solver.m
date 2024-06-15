pkg load symbolic

syms v_c1 v_c1_0 v_c2 v_c2_0 v_c3 v_c3_0 vout vcontrol vref r1 r2 r3 r4 c1 c2 c3 dt i_r1 i_c1 i_c2 i_c3

eqn1 = v_c1 == ((i_r1 + i_c2) * dt / c3 + vref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt) / (1 + c1 / c3 + r2 * c1 / dt);
eqn2 = v_c2 == (( i_c1 + i_c3 - i_r1 - i_c2)* r4 - vout + (r3 * c2 * v_c2_0 / dt)) / (1 + (r3 * c2 / dt));
eqn3 = v_c3 == vcontrol - ( i_c1 + i_c3 - i_r1 - i_c2)* r4;
eqn4 = i_r1 == (((c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - vout) / (r1 + r4));
eqn5 = i_c1 == c1 * (v_c1 - v_c1_0) / dt;
eqn6 = i_c2 == c2 * (v_c2 - v_c2_0) / dt;
eqn7 = i_c3 == c3 * (v_c3 - v_c3_0) / dt;

eqn1 = subs(eqn1, i_c1, c1 * (v_c1 - v_c1_0) / dt);
eqn1 = subs(eqn1, i_c2, c2 * (v_c2 - v_c2_0) / dt);
eqn1 = subs(eqn1, i_c3, c3 * (v_c3 - v_c3_0) / dt);
eqn1 = subs(eqn1, i_r1, (((c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - vout) / (r1 + r4)));
eqn2 = subs(eqn2, i_c1, c1 * (v_c1 - v_c1_0) / dt);
eqn2 = subs(eqn2, i_c2, c2 * (v_c2 - v_c2_0) / dt);
eqn2 = subs(eqn2, i_c3, c3 * (v_c3 - v_c3_0) / dt);
eqn2 = subs(eqn2, i_r1, (((c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - vout) / (r1 + r4)));
eqn3 = subs(eqn3, i_c1, c1 * (v_c1 - v_c1_0) / dt);
eqn3 = subs(eqn3, i_c2, c2 * (v_c2 - v_c2_0) / dt);
eqn3 = subs(eqn3, i_c3, c3 * (v_c3 - v_c3_0) / dt);
eqn3 = subs(eqn3, i_r1, (((c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - vout) / (r1 + r4)));

sol = solve([eqn1, eqn2, eqn3], [v_c1, v_c2, v_c3]);
v_c1 = matlabFunction(factor(sol.v_c1))
v_c2 = matlabFunction(factor(sol.v_c2))
v_c3 = matlabFunction(factor(sol.v_c3))

#eqn1 = (((((c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - vout) / (r1 + r4)) + c2 * (v_c2 - v_c2_0) / dt) * dt / c3 + vref * dt / (r4 * c3) + c1 * v_c1_0 / c3 + v_c3_0 + r2 * c1 * v_c1_0 / dt) / (1 + c1 / c3 + r2 * c1 / dt) == v_c1;

#eqn2 = ((c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - i_r1 - c2 * (v_c2 - v_c2_0) / dt)* r4 - vout + (r3 * c2 * v_c2_0 / dt)) / (1 + (r3 * c2 / dt)) == v_c2;

#eqn3 = vcontrol - (c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - (((c1 * (v_c1 - v_c1_0) / dt + c3 * (v_c3 - v_c3_0) / dt - c2 * (v_c2 - v_c2_0) / dt)* r4 - vout) / (r1 + r4)) - c2 * (v_c2 - v_c2_0) / dt)* r4 == v_c3;

#sol = solve((eqn1, eqn2, eqn3), (v_c1, v_c2, v_c3));

#v_c1 = matlabFunction(factor(sol.v_c1))
#v_c2 = matlabFunction(factor(sol.v_c2))
#v_c3 = matlabFunction(factor(sol.v_c3))
