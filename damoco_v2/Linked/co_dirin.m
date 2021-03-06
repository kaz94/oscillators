function dirin = co_dirin(N1,N2, omeg1,omeg2)
% DAMOCO Toolbox, function CO_DIRIN, version 26.02.14
%
% Given the norms of the coupling functions of two coupled 
% oscillators and their autonomous frequencies, this 
% functions returns the directionality index.
% The index is defined in a way that for symmetrical bidirectional 
% coupling dirin=0 holds, while purely 
% unidirectional coupling 1->2 yields dirin=1;
% unidirectional coupling 2->1 yields dirin=-1.  
%
% Form of call: dirin = co_dirin(N1,N2,omeg1,omeg2)
% Input:        N1,N2 : norms of the coupling functions
%               omeg1,omeg2: frequencies
%
c1=N1/omeg1; % strength of the external contribution to the phase 
c2=N2/omeg2; % dynamics normalized by the natural frequency
% to check for presence of interaction
if c1+c2 < 0.02
    disp('Warning: the coupling is very weak or the systems are not coupled!');
    disp('Result on directionality index may be not reliable');
end
dirin= (c2-c1) / (c1+c2); %  Directionality index
end
