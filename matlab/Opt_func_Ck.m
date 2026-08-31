function [ Sum_Rate, P, Theta ] = Opt_func_Ck( M,N,K,P_max,sigma,C_hat )
% Opt_func_Ck: Alternating Optimization using the CASCADED channel
%   C_hat{k} = diag(conj(h_k)) * G   in C^{M x N},  k = 1..K
% instead of separate H, G. This is algebraically identical to the
% reference Opt_func.m whenever C_hat{k} == diag(H(:,k)')*G exactly
% (verified numerically against Opt_func.m -- see verify_Ck_against_reference.m).
%
% Derivation (see project record):
%   H_all(k,:) = H(:,k)' * Phi * G = theta.' * C_hat{k}     [theta = diag(Phi)]
%   v0 (used in SDR step)          = diag(H(:,k)') * G      = C_hat{k}   (exact match)
%
% Inputs:
%   C_hat : 1xK cell array, each C_hat{k} is (M,N) complex -- estimated (or
%           exact, for verification) cascaded channel per user.
% Outputs: identical meaning/shape to Opt_func.m (Sum_Rate, P, Theta).

%% init
Phi = diag(exp(1j*(rand(M, 1)*2*pi))); % phase shifters
theta_vec = diag(Phi);                  % (M,1)

H_all = compute_H_all(theta_vec, C_hat, K, N);

% ZF
P = H_all'*inv(H_all*H_all');
P = P/norm(P,'fro');

SumRate = [];
iter = 0;

%% Iterations UPDATE
while(1)
iter = iter+1;

Alpha = zeros(K,1);
sum_rate = 0;

for k=1:K
    S_temp = H_all(k,:)*P(:,k);
    S_gain = abs(S_temp)^2;
    IN_gain = 0;

    for j=1:K
        if j==k
            continue;
        end
        IN_temp = H_all(k,:)*P(:,j);
        IN_gain = IN_gain + abs(IN_temp)^2;
    end
    Alpha(k) = S_gain/(IN_gain+sigma);
    sum_rate = sum_rate + log2(1+Alpha(k));
end

SumRate = [SumRate, sum_rate];

Beta = zeros(K,1);

for k=1:K
    S_gain1 = H_all(k,:)*P(:,k);
    IN_gain1 = 0;
    for j=1:K
        IN_temp1 = H_all(k,:)*P(:,j);
        IN_gain1 = IN_gain1 + abs(IN_temp1)^2;
    end
    Beta(k) = sqrt(1+Alpha(k))*S_gain1/(IN_gain1+sigma);
end

mu = diag(sqrt(1+Alpha))*diag(Beta);
T = H_all'*diag(abs(Beta).^2)*H_all;

power = @(dual_v) norm( inv(dual_v*eye(N)+T)*H_all'*mu, 'fro' )^2 - P_max;

low = 1e-6;
high = 50;
tolerance = 1e-5;
dual_v = Bisection(power, low, high, tolerance);

P = inv(dual_v*eye(N)+T)*H_all'*mu;

Theta = conj(diag(Phi));

% V{k,j} = C_hat{k} * P(:,j)   -- exact match to reference's diag(H(:,k)')*G*P(:,j)
V = cell(K,K);
for k=1:K
    v0 = C_hat{k};
    for j=1:K
        v = v0*P(:,j);
        V(k,j) = {v};
    end
end

rho = zeros(K,1);

for k=1:K
    S_gain2 = sqrt(1+Alpha(k))*Theta'*V{k,k};
    IN_gain2 = 0;
    for j=1:K
        IN_temp2 = Theta'*V{k,j};
        IN_gain2 = IN_gain2 + abs(IN_temp2)^2;
    end
    rho(k) = S_gain2/(IN_gain2+sigma);
end

A = zeros(M, M);
b = zeros(M, 1);
for k=1:K
    A = A + ([V{k,:}]*[V{k,:}]')*abs(rho(k))^2;
    b = b + sqrt(1+Alpha(k))*conj(rho(k))*V{k,k};
end

A = (A+A')/2;
cvx_begin sdp quiet
    variable f_SDP
    variable zeta(M)

    maximize(f_SDP - sum(zeta))

    subject to
        zeta >= 0;

        [A + diag(zeta), b; ...
         b', -f_SDP] >= 0;
cvx_end

Theta_opt = inv(A + diag(zeta))*b;
Theta_opt = exp(1i*angle(Theta_opt));  % continuous unit-modulus projection

Phi = diag(Theta_opt');
theta_vec = diag(Phi);

H_all = compute_H_all(theta_vec, C_hat, K, N);

%% if converge
if iter >= 3
    diff = SumRate(iter) - SumRate(iter-1);
    if abs(diff)/SumRate(iter) <= 1e-2
        Sum_Rate = SumRate(end);
        break;
    end
end

end

end


function H_all = compute_H_all(theta_vec, C_hat, K, N)
% H_all(k,:) = theta_vec.' * C_hat{k}   (row vector, 1xN), stacked over k
H_all = zeros(K, N);
for k=1:K
    H_all(k,:) = theta_vec.' * C_hat{k};
end
end