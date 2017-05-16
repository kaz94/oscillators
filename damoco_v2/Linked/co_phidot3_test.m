function [phi1_dot,phi2_dot,phi3_dot,phi1,phi2,phi3] = co_phidot3_test(phi1,phi2,phi3,fsample)
% DAMOCO Toolbox, function CO_PHIDOT3, version 06.03.14
%
% CO_PHIDOT3 computes the derivatives (instantaneous frequencies)  
% of three phases using the Savitzky-Golay filter
% Parameter:   fsample is the sampling frequency
% All phases are truncated to avoid the boundary effect
%
norder=5;   % order of the fitting polynomial
sl=12;      % window semi-length
wl=2*sl+1;  % window length

%%%%%%%%%%%% uncomment for matlab
%[~,g] = sgolay(norder,wl);   % Calculate S-G coefficients
g = load("golay_coeff.m");
phi1=unwrap(phi1); phi2=unwrap(phi2); phi3=unwrap(phi3);

phi1_dot=conv(phi1, g(:,2), 'same');
phi2_dot=conv(phi2, g(:,2), 'same');
phi3_dot=conv(phi3, g(:,2), 'same');
phi1_dot=phi1_dot(sl+1:end-sl)*fsample;
phi2_dot=phi2_dot(sl+1:end-sl)*fsample;
phi3_dot=phi3_dot(sl+1:end-sl)*fsample;
phi1=phi1(sl+1:end-sl); % Truncating both phases in order to 
phi2=phi2(sl+1:end-sl); % synchronize them with the derivative
phi3=phi3(sl+1:end-sl); % synchronize them with the derivative
end

