% run_tau_sweep.m
% Graph 3 driver: runs Estimated-CSI AO (Opt_func_Ck.m, unmodified) for each
% pilot length tau, evaluates TRUE sum rate, computes Effective Spectral
% Efficiency = (1 - tau_total/T_coherence) * SumRate_trueEval, and plots
% EffSE vs pilot overhead.
%
% Prerequisite: run `python export_channels_tau_sweep.py` first (from
% irs_project/) to generate results/tau_sweep/phase2_data_tau*.mat
%
% Requires MATLAB + CVX (cvx_setup already run) -- same dependency as run_phase2.m

clear; close all; clc;

tau_list = [8, 16, 32, 64];
n = length(tau_list);

tau_total_list = zeros(1, n);
sumrate_trueEval_list = zeros(1, n);
effSE_list = zeros(1, n);
T_coherence = 200;  % must match config.system_config.T_coherence

for i = 1:n
    tau = tau_list(i);
    fname = sprintf('results/tau_sweep/phase2_data_tau%d.mat', tau);
    fprintf('\n=== tau=%d (%s) ===\n', tau, fname);
    d = load(fname);

    M = double(d.M); N = double(d.N); K = double(d.K);
    P_max = double(d.P_max); sigma = double(d.sigma);
    H_true = d.H_true; G_true = d.G_true;
    C_hat = d.C_hat;
    T_coherence = double(d.T_coherence);

    rng(1);
    [SumRate_estCk, P_estCk, Theta_estCk] = Opt_func_Ck(M, N, K, P_max, sigma, C_hat);

    % Realistic evaluation: use resulting P, Theta against TRUE channel
    Phi_est = diag(conj(Theta_estCk));
    H_all_true_eval = H_true' * Phi_est * G_true;
    SumRate_trueEval = compute_sum_rate(H_all_true_eval, P_estCk, K, sigma);

    tau_total = double(d.tau_total);
    overhead_frac = tau_total / T_coherence;
    effSE = (1 - overhead_frac) * SumRate_trueEval;

    tau_total_list(i) = tau_total;
    sumrate_trueEval_list(i) = SumRate_trueEval;
    effSE_list(i) = effSE;

    fprintf('tau_total=%d | overhead=%.3f | SumRate(true)=%.4f bps/Hz | EffSE=%.4f bps/Hz\n', ...
        tau_total, overhead_frac, SumRate_trueEval, effSE);
end

save('results/tau_sweep_results.mat', 'tau_list', 'tau_total_list', ...
     'sumrate_trueEval_list', 'effSE_list', 'T_coherence');
fprintf('\nSaved results/tau_sweep_results.mat\n');

%% Plot Graph 3
figure('Position', [100, 100, 700, 500]);
yyaxis left
plot(tau_total_list, effSE_list, 'o-', 'LineWidth', 2, 'MarkerSize', 7);
ylabel('Effective Spectral Efficiency (bps/Hz)');
yyaxis right
plot(tau_total_list, sumrate_trueEval_list, 's--', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('Raw Sum Rate (bps/Hz)');
xlabel('Pilot Overhead \tau_{total} = Q \times \tau  (symbols)');
title('Effective Spectral Efficiency vs Pilot Overhead');
legend('Effective SE (overhead-penalized)', 'Raw Sum Rate (no penalty)', 'Location', 'best');
grid on;
saveas(gcf, 'results/fig3_effSE_vs_overhead.png');
fprintf('Saved results/fig3_effSE_vs_overhead.png\n');


function R = compute_sum_rate(H_all, P, K, sigma)
    R = 0;
    for k = 1:K
        S = abs(H_all(k,:)*P(:,k))^2;
        IN = 0;
        for j = 1:K
            if j==k, continue; end
            IN = IN + abs(H_all(k,:)*P(:,j))^2;
        end
        R = R + log2(1 + S/(IN+sigma));
    end
end
