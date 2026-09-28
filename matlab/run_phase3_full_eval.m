% run_phase3_full_eval.m
% Complete, fair paired Monte Carlo evaluation for Phase 3.
% Evaluates all methods on the EXACT SAME channel and noise realizations.
%
% Methods compared:
%   1. Perfect-CSI AO (Upper benchmark)
%   2. Conventional LS -> Full AO (Converged baseline)
%   3. Conventional LS -> Truncated AO (1 iteration, equal iteration budget)
%   4. Random IRS + RZF (Baseline without IRS optimization)
%   5. Proposed Deep Beamforming (Direct ML, ultra-low latency <5ms)
%   6. Proposed ML + Truncated AO (1 iteration warm-start refinement)

clear; close all; clc;
addpath('matlab');

data_path = 'results/phase3_eval_data.mat';
fprintf('Loading paired test dataset from %s ...\n', data_path);
d = load(data_path);

records = d.records;
n_total = length(records);
noise_levels = d.noise_levels;
n_noise = length(noise_levels);
n_trials = double(d.n_trials);

fprintf('Evaluating %d realizations across %d noise levels (%d trials each)...\n\n', ...
    n_total, n_noise, n_trials);

% Preallocate per-realization results
res_perfect      = zeros(n_total, 1);
res_ls_full      = zeros(n_total, 1);
res_ls_trunc1    = zeros(n_total, 1);
res_rand_rzf     = zeros(n_total, 1);
res_ml_direct    = zeros(n_total, 1);
res_ml_trunc1    = zeros(n_total, 1);

time_ls_full     = zeros(n_total, 1);
iters_ls_full    = zeros(n_total, 1);
time_ls_trunc1   = zeros(n_total, 1);
time_ml_trunc1   = zeros(n_total, 1);
time_ml_direct   = zeros(n_total, 1);

noise_vec        = zeros(n_total, 1);

for i = 1:n_total
    if iscell(records)
        rec = records{i};
    else
        rec = records(i);
    end
    noise_dbm = double(rec.noise_dbm);
    noise_vec(i) = noise_dbm;

    M = double(rec.M);
    N = double(rec.N);
    K = double(rec.K);
    P_max = double(rec.P_max);
    sigma = double(rec.sigma);

    H_true = rec.H_true;
    G_true = rec.G_true;
    C_hat  = rec.C_hat;

    theta_ml = rec.theta_ml(:);
    W_ml     = rec.W_ml;
    t_ml_inf = double(rec.time_ml_infer);
    time_ml_direct(i) = t_ml_inf;

    fprintf('[%2d/%2d] Noise: %5.1f dBm | Trial %2d ... ', ...
        i, n_total, noise_dbm, double(rec.trial) + 1);

    %% 1. Perfect-CSI AO (Upper benchmark)
    rng(1);
    [~, P_perf, Theta_perf] = Opt_func_perfectCSI(M, N, K, P_max, sigma, H_true, G_true);
    Phi_perf = diag(conj(Theta_perf));
    res_perfect(i) = compute_sum_rate(H_true' * Phi_perf * G_true, P_perf, K, sigma);

    %% 2. Conventional LS -> Full AO (Main baseline)
    rng(1);
    t0 = tic;
    [~, P_ls_full, Theta_ls_full, it_ls] = Opt_func_Ck(M, N, K, P_max, sigma, C_hat);
    time_ls_full(i) = toc(t0);
    iters_ls_full(i) = it_ls;
    Phi_ls_full = diag(conj(Theta_ls_full));
    res_ls_full(i) = compute_sum_rate(H_true' * Phi_ls_full * G_true, P_ls_full, K, sigma);

    %% 3. Conventional LS -> Truncated AO (1 iteration)
    rng(1);
    t0 = tic;
    [~, P_ls_tr1, Theta_ls_tr1] = Opt_func_Ck(M, N, K, P_max, sigma, C_hat, [], 1);
    time_ls_trunc1(i) = toc(t0);
    Phi_ls_tr1 = diag(conj(Theta_ls_tr1));
    res_ls_trunc1(i) = compute_sum_rate(H_true' * Phi_ls_tr1 * G_true, P_ls_tr1, K, sigma);

    %% 4. Random IRS + RZF Baseline
    rng(42 + i);
    theta_rand = exp(1i * rand(M, 1) * 2 * pi);
    Phi_rand = diag(conj(theta_rand));
    H_eff_rand = zeros(K, N);
    for k = 1:K
        H_eff_rand(k, :) = theta_rand.' * C_hat{k};
    end
    W_rzf_unnorm = H_eff_rand' * inv(H_eff_rand * H_eff_rand' + (K * sigma / P_max) * eye(K));
    for k = 1:K
        W_rzf_unnorm(:, k) = W_rzf_unnorm(:, k) / norm(W_rzf_unnorm(:, k));
    end
    P_rzf = W_rzf_unnorm * sqrt(P_max / K);
    res_rand_rzf(i) = compute_sum_rate(H_true' * Phi_rand * G_true, P_rzf, K, sigma);

    %% 5. Proposed Deep Beamforming (Direct ML)
    Phi_ml = diag(conj(theta_ml));
    res_ml_direct(i) = compute_sum_rate(H_true' * Phi_ml * G_true, W_ml, K, sigma);

    %% 6. Proposed ML + Truncated AO (1 iteration refinement)
    t0 = tic;
    [~, P_ml_tr1, Theta_ml_tr1] = Opt_func_Ck(M, N, K, P_max, sigma, C_hat, theta_ml, 1);
    time_ml_trunc1(i) = toc(t0) + t_ml_inf;
    Phi_ml_tr1 = diag(conj(Theta_ml_tr1));
    res_ml_trunc1(i) = compute_sum_rate(H_true' * Phi_ml_tr1 * G_true, P_ml_tr1, K, sigma);

    fprintf('LS: %5.2f | ML-Dir: %5.2f | ML+AO(1): %5.2f | Perf: %5.2f bps/Hz\n', ...
        res_ls_full(i), res_ml_direct(i), res_ml_trunc1(i), res_perfect(i));
end

%% Aggregate results per noise level
summary_rows = [];
fprintf('\n========================================================================================\n');
fprintf('FINAL COMPARATIVE RESULTS TABLE (True-Channel Achievable Sum-Rate in bps/Hz)\n');
fprintf('========================================================================================\n');
fprintf('%6s | %10s | %10s | %10s | %10s | %10s | %8s | %8s\n', ...
    'Noise', 'Rand+RZF', 'LS Full AO', 'LS AO(1)', 'ML Direct', 'ML+AO(1)', 'Gain(ML-LS)', 'WinRate');
fprintf('----------------------------------------------------------------------------------------\n');

results_struct = struct();

for n = 1:n_noise
    noise_dbm = noise_levels(n);
    mask = (noise_vec == noise_dbm);

    mean_perf   = mean(res_perfect(mask));
    mean_rand   = mean(res_rand_rzf(mask));
    mean_ls     = mean(res_ls_full(mask));
    med_ls      = median(res_ls_full(mask));
    std_ls      = std(res_ls_full(mask));

    mean_ls_tr1 = mean(res_ls_trunc1(mask));
    mean_ml_dir = mean(res_ml_direct(mask));
    med_ml_dir  = median(res_ml_direct(mask));
    std_ml_dir  = std(res_ml_direct(mask));

    mean_ml_tr1 = mean(res_ml_trunc1(mask));
    med_ml_tr1  = median(res_ml_trunc1(mask));

    gain_dir    = mean_ml_dir - mean_ls;
    gain_tr1    = mean_ml_tr1 - mean_ls;
    win_rate    = mean(res_ml_direct(mask) > res_ls_full(mask)) * 100;
    win_rate_tr = mean(res_ml_trunc1(mask) > res_ls_trunc1(mask)) * 100;

    fprintf('%5.1f  | %10.2f | %10.2f | %10.2f | %10.2f | %10.2f | %+8.2f   | %6.1f%%\n', ...
        noise_dbm, mean_rand, mean_ls, mean_ls_tr1, mean_ml_dir, mean_ml_tr1, gain_dir, win_rate);

    key = sprintf('noise_%s', strrep(num2str(abs(noise_dbm)), '.', 'p'));
    results_struct.(key).noise_dbm = noise_dbm;
    results_struct.(key).mean_perfect = mean_perf;
    results_struct.(key).mean_rand_rzf = mean_rand;
    results_struct.(key).mean_ls_full = mean_ls;
    results_struct.(key).median_ls_full = med_ls;
    results_struct.(key).std_ls_full = std_ls;
    results_struct.(key).mean_ls_trunc1 = mean_ls_tr1;
    results_struct.(key).mean_ml_direct = mean_ml_dir;
    results_struct.(key).median_ml_direct = med_ml_dir;
    results_struct.(key).std_ml_direct = std_ml_dir;
    results_struct.(key).mean_ml_trunc1 = mean_ml_tr1;
    results_struct.(key).median_ml_trunc1 = med_ml_tr1;
    results_struct.(key).abs_gain = gain_dir;
    results_struct.(key).win_rate = win_rate;
    results_struct.(key).win_rate_trunc1 = win_rate_tr;
end

fprintf('========================================================================================\n');
fprintf('AVERAGE RUNTIMES:\n');
fprintf('  Conventional LS -> Full AO: %6.2f s/realization (avg %3.1f iterations)\n', ...
    mean(time_ls_full), mean(iters_ls_full));
fprintf('  Conventional LS -> AO(1):   %6.2f s/realization (1 iteration)\n', mean(time_ls_trunc1));
fprintf('  Proposed Deep Beamforming:  %6.4f s/realization (Inference only, ZERO iterations)\n', mean(time_ml_direct));
fprintf('  Proposed ML + AO(1):        %6.2f s/realization (ML + 1 iteration)\n', mean(time_ml_trunc1));
fprintf('  Speedup of Direct ML over Full AO: %.1fx\n', mean(time_ls_full) / max(1e-4, mean(time_ml_direct)));
fprintf('========================================================================================\n');

% Save detailed per-realization arrays and summary
save('results/phase3_evaluation_results.mat', ...
    'noise_vec', 'res_perfect', 'res_rand_rzf', 'res_ls_full', 'res_ls_trunc1', ...
    'res_ml_direct', 'res_ml_trunc1', 'time_ls_full', 'iters_ls_full', ...
    'time_ls_trunc1', 'time_ml_direct', 'time_ml_trunc1', 'results_struct');
fprintf('\nSaved results/phase3_evaluation_results.mat\n');

% Export CSV summary
csv_file = 'results/phase3_evaluation_summary.csv';
fid = fopen(csv_file, 'w');
fprintf(fid, 'noise_dbm,rand_rzf_mean,ls_full_mean,ls_full_med,ls_trunc1_mean,ml_direct_mean,ml_direct_med,ml_trunc1_mean,gain_ml_vs_ls,win_rate_percent,avg_time_ls_full,avg_time_ml_direct\n');
for n = 1:n_noise
    noise_dbm = noise_levels(n);
    mask = (noise_vec == noise_dbm);
    fprintf(fid, '%.1f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.4f,%.6f\n', ...
        noise_dbm, mean(res_rand_rzf(mask)), mean(res_ls_full(mask)), median(res_ls_full(mask)), ...
        mean(res_ls_trunc1(mask)), mean(res_ml_direct(mask)), median(res_ml_direct(mask)), ...
        mean(res_ml_trunc1(mask)), mean(res_ml_direct(mask)) - mean(res_ls_full(mask)), ...
        mean(res_ml_direct(mask) > res_ls_full(mask)) * 100, ...
        mean(time_ls_full(mask)), mean(time_ml_direct(mask)));
end
fclose(fid);
fprintf('Saved summary CSV to %s\n', csv_file);


function R = compute_sum_rate(H_all, P, K, sigma)
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
