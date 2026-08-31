% verify_Ck_against_reference.m
% Verifies that Opt_func_Ck.m, given EXACT (noiseless) C_hat{k} = diag(H(:,k)')*G,
% produces IDENTICAL Sum_Rate/P/Theta to the original Opt_func.m.
% Run this once, standalone, before trusting Opt_func_Ck.m for Phase 2 results.
%
% Requires: MATLAB + CVX installed and set up (cvx_setup already run).

clear; close all; clc;
rng(42);  % reproducibility

%% test system (matches Phase 2 default config dimensions -- small sizes like
%% M=8,N=4,K=2 were found to be numerically degenerate: H_all becomes near-
%% singular and the CVX SDP becomes ill-posed. Use realistic-scale dimensions.
M = 64;  % IRS elements
N = 8;   % BS antennas
K = 4;   % users
P_max = 1e-6;
sigma = 3.16e-12;

H = 1e-3*(randn(M,K) + 1i*randn(M,K))/sqrt(2);
G = 1e-3*(randn(M,N) + 1i*randn(M,N))/sqrt(2);

% Build exact C_hat{k} from true H, G (noiseless -- this is the verification case)
C_hat = cell(1,K);
for k=1:K
    C_hat{k} = diag(H(:,k)')*G;
end

%% run original (perfect-CSI) AO
rng(1);  % same random init phase for both runs -- must reseed identically
[Sum_Rate_orig, P_orig, Theta_orig] = Opt_func_perfectCSI(M,N,K,P_max,sigma,H,G);

%% run C_k-based AO with exact C_hat
rng(1);  % identical seed for identical random init
[Sum_Rate_Ck, P_Ck, Theta_Ck] = Opt_func_Ck(M,N,K,P_max,sigma,C_hat);

%% compare
fprintf('Sum_Rate original : %.10f\n', Sum_Rate_orig);
fprintf('Sum_Rate Ck-based  : %.10f\n', Sum_Rate_Ck);
fprintf('Sum_Rate abs diff  : %.2e\n', abs(Sum_Rate_orig - Sum_Rate_Ck));

fprintf('P relative error   : %.2e\n', norm(P_orig - P_Ck,'fro')/norm(P_orig,'fro'));
fprintf('Theta relative err : %.2e\n', norm(Theta_orig - Theta_Ck)/norm(Theta_orig));

tol = 1e-6;
assert(abs(Sum_Rate_orig - Sum_Rate_Ck) < tol, 'Sum_Rate MISMATCH -- Ck substitution has a bug.');
assert(norm(P_orig-P_Ck,'fro')/norm(P_orig,'fro') < tol, 'P MISMATCH.');
assert(norm(Theta_orig-Theta_Ck)/norm(Theta_orig) < tol, 'Theta MISMATCH.');

 fprintf('\nPASS: Opt_func_Ck.m with exact C_hat reproduces Opt_func.m exactly.\n');