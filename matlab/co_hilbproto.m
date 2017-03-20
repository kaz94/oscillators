function [theta,minampl]= co_hilbproto(x,fignum,x0,y0,ntail)
% DAMOCO Toolbox, function CO_HILBPROTO, version 27.02.14
% CO_HILBPROTO computes the prophase via the Hilbert transform
% The output is truncated from both ends to avoid boundary effects
%
% Form of call: [theta,minampl]= co_hilbphase(x,fignum,x0,y0,ntail)
%               [theta,minampl]= co_hilbphase(x,fignum,x0,y0)
%               [theta,minampl]= co_hilbphase(x,fignum,x0)
%               [theta,minampl]= co_hilbphase(x,fignum)
%               [theta,minampl]= co_hilbphase(x)
%
% INPUT:  x      is scalar timeseries,
%         x0,y0  are coordinates of the origin (by default x0=0, y0=0)
%         ntail  is the number of points at the ends to be cut off,
%                by default ntail=1000
% Output: theta is the protophase in 0,2pi interval
%         minamp is the minimal instantaneous amplitude
%
if nargin < 5, ntail=1000; end
if nargin < 4, y0=0.0;     end
if nargin < 3, x0=0;       end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ht=hilbert(x); 
ht=ht(ntail+1:end-ntail);
mean(ht)
ht=ht-mean(ht);  % subtracting the mean value    

%ht=ht-x0-1i*y0;    
theta=angle(ht);
theta=mod(theta,2*pi);  % phase is in 0,2pi interval
plot(theta)
theta=theta(:);         % to ensure that theta is a column vector
end