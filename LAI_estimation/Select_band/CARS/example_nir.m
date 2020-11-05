%+++ Import data
load wheat_m51
%+++ Choose the number of latent variables
MCCV=plsmccv(X,y,25,'center',1000,0.8);

%+++ Using CARS to perform variable selection
CARS=carspls(X,y,MCCV.optPC,5,'center',50); 
plotcars(CARS);
SelectedVariables=CARS.vsel


%+++ Also the Simplified version of CARS could also be used. This programe
%+++ scarspls.m can reproduce the variable selection results 
sCARS=scarspls(X,y,MCCV.optPC,5,'center',50); 
plotcars(sCARS);
SelectedVariables=sCARS.vsel




