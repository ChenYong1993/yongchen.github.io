function [A,X,S,Out] = LRNMF_HSI_denoise(Y,opts)
%% Initiation
maxit = opts.maxit;
tol = opts.tol;              
A = opts.A;
X = opts.X;
rho1 = opts.rho1;  
rho2 = opts.rho2;
lambda1 = opts.lambda1;    
x_size = opts.x_size;            
frame = opts.frame;           
Level = opts.Level;               
beta1= opts.beta1;                     
S = zeros(size(Y));   
%%
X_p = X;
A_p = A;
Out.Rse = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
[D,~] = GenerateFrameletFilter(frame);
m = x_size(1);
n = x_size(2);
e = size(X,1);
X_transition = zeros(m,n*e);
Z = FraDecMultiLevel(X_transition,D,Level);
Theta = Z;
Theta2 = zeros(size(X));
U = zeros(size(X));
LA = zeros(size(A));
AD = zeros(size(A));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for it = 1:maxit
    % update A
    A = ((Y - S) * (X)' + rho1 * A + beta1*(AD - LA)) * pinv(X * X' + (rho1+beta1) * eye(size(X * X')));
    AD = max(A+LA,0);
    LA = LA + (A - AD);
    % update S
    W = ((Y - A * X) + rho2 * S)./(1 + rho2);
    S = sign(W).* max( abs(W) - lambda1/(1+rho2), 0);
    % update X
    D = Y - S;
    [Z,Theta,U,Theta2,X] = Framelet_X(A,D,X,opts,Z,Theta,U,Theta2);
    %%
    Y1 = A*X;
    Y2 = A_p*X_p;
    Rel_Err = norm(Y1 - Y2,'fro')/norm(Y1,'fro');
    A_p = A;
    X_p = X;
    if it > 2
    Out.Rse = [Out.Rse;Rel_Err];  
    end
    if Rel_Err < tol  
        break;
    end    
end