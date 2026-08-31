% run_phase2.m
% Phase 2 driver: runs BOTH classical baselines on the same channel realization
% and saves results for comparison / MATLAB plotting.
%
%   1. Perfect-CSI AO   (Opt_func_perfectCSI.m, UNMODIFIED reference algorithm)
%      -- upper bound, not a realistic deployable baseline.
%   2. Estimated-CSI AO (Opt_func_Ck.m, uses C_hat from multi-phase noisy pilots)
%      -- the realistic classical baseline for comparison against the GNN (Phase 3).
%
% Prerequisite: run `python -m matlab.export_channels` first (from irs_project/)
% to generate results/phase2_data_*.mat
%
% Requires MATLAB + CVX (cvx_setup already run).

clear; close all; clc;

data_files = {'results/phase2_data_good_snr.mat', 'results/phase2_data_mid_snr.mat'};
results = struct();

for f = 1:length(data_files)
    fname = data_files{f};
    fprintf('\n=== Processing %s ===\n', fname);
    d = load(fname);

    M = double(d.M); N = double(d.N); K = double(d.K);
    P_max = double(d.P_max); sigma = double(d.sigma);
    H_true = d.H_true; G_true = d.G_true;
    C_hat = d.C_hat;  % 1xK cell, loaded from .mat object array

    %% 1. Perfect-CSI AO (unmodified reference, upper bound)
    rng(1);
    [SumRate_perfect, P_perfect, Theta_perfect] = ...
        Opt_func_perfectCSI(M,N,K,P_max,sigma,H_true,G_true);
    fprintf('Perfect-CSI AO Sum Rate      : %.4f bps/Hz\n', SumRate_perfect);

    %% 2. Estimated-CSI AO (realistic baseline, uses noisy-pilot-derived C_hat)
    rng(1);
    [SumRate_estCk, P_estCk, Theta_estCk] = ...
        Opt_func_Ck(M,N,K,P_max,sigma,C_hat);

    % IMPORTANT: the algorithm only SEES C_hat (estimated). But real-world
    % performance must be judged on the TRUE channel -- recompute sum-rate
    % using the resulting P, Theta against H_true, G_true.
    Phi_est = diag(conj(Theta_estCk));
    H_all_true_eval = H_true' * Phi_est * G_true;
    SumRate_estCk_trueEval = compute_sum_rate(H_all_true_eval, P_estCk, K, sigma);

    fprintf('Estimated-CSI AO Sum Rate (as seen by algorithm, using C_hat): %.4f bps/Hz\n', SumRate_estCk);
    fprintf('Estimated-CSI AO Sum Rate (TRUE, realistic eval)             : %.4f bps/Hz\n', SumRate_estCk_trueEval);
    fprintf('Pilot overhead: tau_total = %d symbols (Q=%d phases)\n', d.tau_total, d.Q);
    fprintf('C_hat recovery error (verification only): %s\n', mat2str(d.recovery_err, 4));

    key = sprintf('run%d', f);
    results.(key).file = fname;
    results.(key).SumRate_perfect = SumRate_perfect;
    results.(key).SumRate_estCk_algSeen = SumRate_estCk;
    results.(key).SumRate_estCk_trueEval = SumRate_estCk_trueEval;
    results.(key).tau_total = d.tau_total;
    results.(key).recovery_err = d.recovery_err;
    results.(key).pilot_noise_power_dbm = d.pilot_noise_power_dbm;
end

save('results/phase2_results.mat', 'results');
fprintf('\nSaved results/phase2_results.mat\n');


function R = compute_sum_rate(H_all, P, K, sigma)
    R = 0;
    for k=1:K
        S = abs(H_all(k,:)*P(:,k))^2;
        IN = 0;
        for j=1:K
            if j==k, continue; end
            IN = IN + abs(H_all(k,:)*P(:,j))^2;
        end
        R = R + log2(1 + S/(IN+sigma));
    end
end