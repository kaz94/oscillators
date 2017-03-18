data=load("../signal.txt");
t=data(:,1);
x1=data(:,2);
y1=data(:,3);
x2=data(:,4);
y2=data(:,5);
x3=data(:,6);
y3=data(:,7);

pkg load signal
[theta1,minampl]= co_hilbproto(x1, 0, 0,0, ntail=4000);
[theta2,minampl]= co_hilbproto(x2, 0, 0,0, ntail=4000);
[theta3,minampl]= co_hilbproto(x3, 0, 0,0, ntail=4000);




%{
theta=[theta1 theta2 theta3];
size(theta)
save /home/kasia/PycharmProjects/oscillators/matlab/theta_mat.txt theta
%}
nfft=10;
[phi1,arg,sigma] = co_fbtransf1(theta1,nfft);
[phi2,arg,sigma] = co_fbtransf1(theta2,nfft);
[phi3,arg,sigma] = co_fbtransf1(theta3,nfft);

phi=[phi1 phi2 phi3];
size(phi)
save phi_mat_moje.txt phi

%x=[x1 x2 x3];
%hb=real(hilbert(x));
%save /home/kasia/PycharmProjects/oscillators/matlab/hilbert.txt hb

%python = load("/home/kasia/PycharmProjects/oscillators/true.txt");
%x_py = python(:,1);

%plot(1:length(theta1), theta1,'r',1:length(theta1), phi1,'g',1:length(x_py), x_py+pi, 'b')
%plot(1:length(phi1), phi1, color='r')
