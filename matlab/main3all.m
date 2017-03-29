% Reconstructing phase dynamics of oscillator networks
% Kralemann, Bjorn Pikovsky, Arkady S. Rosenblum, Michael G.

sig = zeros(3);
sig(1, 2) = 1;
sig(2, 3) = 1;
sig(3, 1) = 1;
eps = 0.1;
eta = 0.0;
mu = 0.5;
om = [1, 1.32347, 1.75483];

% vanderpol is defined below
ff = @(t, y) vanderpol(t, y, mu, om, eps, eta, sig);

% duffing is defined below
% d = 1;
% e = -1;
% ff = @(t, y) duffing(t, y, mu, d, e, om, eps, eta, sig);

[t, y] = ode23s(ff,[0 10000],[0.1; 0.1; 0.1; 0.1; 0.1; 0.1]);

t = t(10000:end);
y = y(10000:end, :);
fs = 1/(t(2) - t(1));

x1 = y(:, 1);
y1 = y(:, 2);
x2 = y(:, 3);
y2 = y(:, 4);
x3 = y(:, 5);
y3 = y(:, 6);

% protopase
th1 = co_hilbproto(x1, -1, 0, 0, 500);
th2 = co_hilbproto(x2, -1, 0, 0, 500);
th3 = co_hilbproto(x3, -1, 0, 0, 500);

th1 = -atan2(x1,y1);
th2 = -atan2(x2,y2);
th3 = -atan2(x3,y3);

% genuine phase
wphi1 = co_fbtrT(th1);
wphi2 = co_fbtrT(th2);
wphi3 = co_fbtrT(th3);

% angular freq
[Dphi1, Dphi2, Dphi3, phi1, phi2, phi3] = co_phidot3(wphi1, wphi2, wphi3, fs);

% triplet analysis of a network with 3 oscillators
[Qcoef1, Qcoef2, Qcoef3] = co_fcpltri(phi1, phi2, phi3, Dphi1, Dphi2, Dphi3, 5);
[COUP, NORM, OMEGA] = co_tricplfan(Qcoef1, Qcoef2, Qcoef3, 1)
% [COUP] = co_nettri([phi1, phi2, phi3], [Dphi1, Dphi2, Dphi3], 5, 1);
co_plottri(1, COUP)

function dydt = vanderpol(t, y, mu, om, eps, eta, sig)
    dydt(1) = y(2);
    dydt(2) = mu * (1.0 - y(1)^2) * y(2) - om(1)^2 * y(1) + ...
         eps * (sig(1, 2) * (y(3) + y(4)) + sig(1, 3) * (y(5) + y(6))) + ...
         sig(1, 1) * eta * y(3)*y(5);
    dydt(3) = y(4);
    dydt(4) = mu * (1.0 - y(3)^2) * y(4) - om(2)^2 * y(3) + ...
         eps * (sig(2, 1) * (y(1) + y(2)) + sig(2, 3) * (y(5) + y(6))) + ...
         sig(2, 2) * eta * y(1)*y(5);
    dydt(5) = y(6);
    dydt(6) = mu * (1.0 - y(5)^2) * y(6) - om(3)^2 * y(5) + ...
         eps * (sig(3, 1) * (y(1) + y(2)) + sig(3, 2) * (y(3) + y(4))) + ...
         sig(3, 3) * eta * y(1)*y(3);
    dydt = dydt';
end

function dydt = duffing(t, y, mu, d, e, om, eps, eta, sig)
    dydt(1) = y(2);
    dydt(2) = mu * (1.0 - y(1)^2) * y(2) - om(1)^2 * y(1)  * (y(1) + d) * (y(1) + e) + ...
         eps * (sig(1, 2) * (y(3) + y(4)) + sig(1, 3) * (y(5) + y(6))) + ...
         sig(1, 1) * eta * y(3)*y(5);
    dydt(3) = y(4);
    dydt(4) = mu * (1.0 - y(3)^2) * y(4) - om(2)^2 * y(3) * (y(3) + d) * (y(3) + e) + ...
         eps * (sig(2, 1) * (y(1) + y(2)) + sig(2, 3) * (y(5) + y(6))) + ...
         sig(2, 2) * eta * y(1)*y(5);
    dydt(5) = y(6);
    dydt(6) = mu * (1.0 - y(5)^2) * y(6) - om(3)^2 * y(5) * (y(5) + d) * (y(5) + e) + ...
         eps * (sig(3, 1) * (y(1) + y(2)) + sig(3, 2) * (y(3) + y(4))) + ...
         sig(3, 3) * eta * y(1)*y(3);
    dydt = dydt';
end

