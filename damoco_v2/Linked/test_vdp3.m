clear all;
pkg load signal
pkg load odepkg
%M=load('time_series.dat');
M=load('vdp3.dat');
time=M(:,1); 
fs=1/(time(2)-time(1));
%th1=-atan2(M(:,3),M(:,2));
%th2=-atan2(M(:,5),M(:,4));
%th3=-atan2(M(:,7),M(:,6));
th1 = M(:,2);
th2 = M(:,3);
th3 = M(:,4);
clear M;

phi1 = co_fbtrT(th1);
phi2 = co_fbtrT(th2);
phi3 = co_fbtrT(th3);
[M_SyncIn,maxind,n_ph1,m_ph2,l_ph3]=co_maxsync3(phi1,phi2,phi3,5);
disp(maxind)
disp(n_ph1)
disp(m_ph2)
disp(l_ph3)

%figure(1); plot(time,th1,time,th2,time,th3); 
%figure(2); plot(time,phi1,time,phi2,time,phi3);
%figure(3); hist(th1,100); figure(4); hist(phi1,100);

[Dphi1, Dphi2, Dphi3, phi1, phi2, phi3] = co_phidot3_test(phi1, phi2, phi3, fs);
%{
Dphi1 = diff(phi1) / fs;
Dphi2 = diff(phi2) / fs;
Dphi3 = diff(phi3) / fs;
phi1 = phi1(1:end-1);
phi2 = phi2(1:end-1);
phi3 = phi3(1:end-1);
%}
%figure(5); plot(Dphi3);
%[Qcoef1, Qcoef2, Qcoef3] = co_fcpltri(phi1, phi2, phi3, Dphi1, Dphi2, Dphi3, 5);
%[COUP] = co_nettri([phi1, phi2, phi3], [Dphi1, Dphi2, Dphi3], 5, 1)
[Qcoef1, Qcoef2, Qcoef3] = co_fcpltri(phi1, phi2, phi3, Dphi1, Dphi2, Dphi3, 5);
[COUP, NORM, OMEGA] = co_tricplfan(Qcoef1, Qcoef2, Qcoef3, 1)
