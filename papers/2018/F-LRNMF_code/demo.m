%% Framelet
clear all;
close all;
clc;	
path(path,genpath(pwd));
%% simulated experiment
%----------------------------------------------load image---------------------------------------------------------
rng(1)
load HSI_Noi
load WDC
OriData3 = WDC;
%----------------------------------------------noise simulated---------------------------------------------------------
[M,N,p] = size(OriData3);
oriData3_noise = Noi_H;
%% parameters       %different images and noise levels may need change parameters slightly.
Lambda1 = 0.05;
Lambda2 = 0.3;
Rho = 0.1;
%%%%%  Initialization 
c = 9;                      
A_init = rand(p,c);
X_init = rand(c,M*N);
Nway = [M,N,c];
%% %% 
opts = [];
opts.maxit = 120;
opts.c = c;    
opts.lambda1 = Lambda1;        
opts.lambda2 = Lambda2;         
opts.rho1 = Rho;
opts.rho2 = Rho;
opts.rho3 = Rho;   
opts.beta1 = 200*opts.lambda2;              
opts.beta2 = 200*opts.lambda2;              
opts.A = A_init;
opts.X = X_init;
opts.tol = 1e-4;
opts.frame = 1;
opts.Level = 1;  
opts.F_it = 1;
opts.wLevel = 1/2;
opts.x_size = Nway(1:2);
%% 
Nhsi = reshape(oriData3_noise,[],p)';
disp('Begin the NMFFramelet algorithm')
tic
[A,X,S,Out ] = LRNMF_HSI_denoise(Nhsi,opts);
toc
%%
Y_re = A*X;
Y_re = reshape(Y_re',M,N,p);
[psnr, ssim] = MSIQA(OriData3 * 255, Y_re  * 255);
