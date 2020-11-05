clear;
load('e:\Program Files\MATLAB71\work\pls\corn_m51.mat');
%%%%%%%%

method='none';
[X]=pretreat(X,method);
y=pretreat(y,method);

[Mx,Nx]=size(X);
[B,Wstar,T,P,Q,R2X,R2Y,W]=pls_nipals(X,y,min(size(X)),0);
U=zeros(Mx,min(size(X)));
for i=1:size(T,2);U(:,i)=T(:,i)/norm(T(:,i));end

R=U'*X*W
     
%+++ X truncation
Atrunc=4;
% U truncation
XBrecoverU=U(:,1:Atrunc)*R(1:Atrunc,:)*W'
% U&W truncation
XDrecover=U(:,1:Atrunc)*R(1:Atrunc,1:(Atrunc+1))*W(:,(1:Atrunc+1))'
% W truncation
XBrecoverW=U*R(:,1:Atrunc)*W(:,1:Atrunc)'
% Conventional-PLS 
XCrecover=T(:,1:Atrunc)*P(:,1:Atrunc)'

EBU=XBrecoverU-X;
EBW=XBrecoverW-X;
EC=XCrecover-X;

%+++ plot
% ID=Atrunc+1;
% subplot(211);
% plot(XCrecover(:,ID),[XBrecoverU(:,ID) XBrecoverW(:,ID)]);
% xlabel('C-PLS');
% legend('U\_truncation','W\_truncation',4);
% subplot(212);
% plot(XCrecover(:,ID),[XBrecoverU(:,ID)-XCrecover(:,ID)]);
% R(Atrunc,Atrunc+1)

%+++ Spectra reconstruction
ID=1;
plot([X(ID,:)'],'b');
hold on;
plot(XBrecoverU(ID,:),'g')
plot(XBrecoverW(ID,:)','r')
plot(XCrecover(ID,:)','k' )
plot(XDrecover(ID,:)','c' )

