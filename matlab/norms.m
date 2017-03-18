
Qcoef = load("qc1py.txt");
S=size(Qcoef)
N = (S(1)-1) / 2;
omega = real(Qcoef(N+1,N+1));            % defining estimate of natural frequency as Qcoef_0_0
Qcoef(N+1,N+1)=0;                        % Setting the constant term, omega, to zero  
Nrmq = sqrt(trapz(trapz (abs(Qcoef).^2)));   % Computing the norm of coupling function 
