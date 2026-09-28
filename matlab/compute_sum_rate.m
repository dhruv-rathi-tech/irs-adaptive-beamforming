function R = compute_sum_rate(H_all, P, K, sigma)
% compute_sum_rate: Computes the downlink achievable sum-rate in bps/Hz.
% Inputs:
%   H_all : (K, N) complex effective channel matrix
%   P     : (N, K) complex beamforming matrix
%   K     : number of users
%   sigma : noise variance in Watts
    R = 0;
    for k = 1:K
        S = abs(H_all(k, :) * P(:, k))^2;
        IN = 0;
        for j = 1:K
            if j == k, continue; end
            IN = IN + abs(H_all(k, :) * P(:, j))^2;
        end
        R = R + log2(1 + S / (IN + sigma));
    end
end
