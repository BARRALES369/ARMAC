%--------------------------------------------------------------------------
% Choix du point initial (vecteur colonne)
clear all
close all
X0 = [0 0]';

%--------------------------------------------------------------------------
% Configuration de l'algorithme d'optimisation

tol = 0.01;             % Critere d'arret norm(grad) < tol
maxit = 10;            % Nombre d'iteration maximum
budget = 20;           % Nombre maximum d'?valuations
stepsize = 1e-9;       % Taille du pas pour l'estimation du gradient

%--------------------------------------------------------------------------
% Appel de l'algorithme d'optimisation

[X,histout] = steep(X0,@rosenbrock,tol,maxit,budget);

% [X,histout] = nelder(simplex(X0,0.1),@rosenbrock,tol,maxit,budget);
% [X,histout] = bfgs(X0,@rosenbrock,tol,maxit,budget);

% [X,histout] = bfgs(X0,@rosenbrock,tol,maxit,budget,stepsize);

%--------------------------------------------------------------------------
% Affichage graphique de la solution dans la figure 1
figure(1);
plotRosenbrock(X);

%--------------------------------------------------------------------------
% Affichage de l'evolution de la fonction objectif et de la norme du 
% gradient au cours des it?rations dans la figure 2
figure(2);
plotHist(histout);