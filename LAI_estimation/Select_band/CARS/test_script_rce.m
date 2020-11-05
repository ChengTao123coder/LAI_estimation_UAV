%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%   MLR for removing the effects of confounding factors  %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
load RCE_example

X=data(:,[1 2 3 4 5]);
Y=data(:,6:end);
CIndex=[3];
R=rce(X,Y,5,'autoscaling',CIndex)

%+++ PCA plot using the original metabolic profiles and samples are grouped
% according to the sex.
sex=X(:,3);
[U,S,V]=csvd(Y);
classplot2(U(:,1:2),,0);




















