function run_eval_range(start_idx, end_idx, part_id)
% run_eval_range: Evaluates a subset [start_idx, end_idx] of the 500-realization
% independent test dataset.
%
% Evaluates all 6 methods:
%   1. Perfect-CSI AO (Upper benchmark)
%   2. Conventional LS -> Full AO (Main baseline)
%   3. Conventional LS -> Truncated AO (1 iter)
%   4. Random IRS + RZF
%   5. Proposed Deep Beamforming (Direct ML)
%   6. Proposed ML + Truncated AO (1 iter warm-start)
%
% Saves periodic checkpoints every 5 realizations.
% Supports automatic resume if interrupted.

addpath('matlab');

data_path = 'results/phase3_independent_eval_data.mat';
ckpt_path = sprintf('results/phase3_independent_checkpoint_part%d.mat', part_id);
final_path = sprintf('results/phase3_independent_results_part%d.mat', part_id);

fprintf('[Worker %d] Loading dataset from %s ...\n', part_id, data_path);
d = load(data_path);
records = d.records;

n_sub = end_idx - start_idx + 1;
fprintf('[Worker %d] Target range: %d to %d (%d realizations)\n', ...
    part_id, start_idx, end_idx, n_sub);

% Preallocate arrays
seeds          = zeros(n_sub, 1);
noise_vec      = zeros(n_sub, 1);
trials         = zeros(n_sub, 1);

res_perfect    = zeros(n_sub, 1);
res_ls_full    = zeros(n_sub, 1);
res_ls_trunc1  = zeros(n_sub, 1);
res_rand_rzf   = zeros(n_sub, 1);
res_ml_direct  = zeros(n_sub, 1);
res_ml_trunc1  = zeros(n_sub, 1);

time_ls_full   = zeros(n_sub, 1);
iters_ls_full  = zeros(n_sub, 1);
time_ls_trunc1 = zeros(n_sub, 1);
time_ml_trunc1 = zeros(n_sub, 1);
time_ml_direct = zeros(n_sub, 1);
time_ls_est    = zeros(n_sub, 1);

completed_mask = false(n_sub, 1);

% Check for existing checkpoint to resume
if exist(ckpt_path, 'file')
    fprintf('[Worker %d] Found existing checkpoint %s, resuming...\n', part_id, ckpt_path);
    ckpt = load(ckpt_path);
    seeds          = ckpt.seeds;
    noise_vec      = ckpt.noise_vec;
    trials         = ckpt.trials;
    res_perfect    = ckpt.res_perfect;
    res_ls_full    = ckpt.res_ls_full;
    res_ls_trunc1  = ckpt.res_ls_trunc1;
    res_rand_rzf   = ckpt.res_rand_rzf;
    res_ml_direct  = ckpt.res_ml_direct;
    res_ml_trunc1  = ckpt.res_ml_trunc1;
    time_ls_full   = ckpt.time_ls_full;
    iters_ls_full  = ckpt.iters_ls_full;
    time_ls_trunc1 = ckpt.time_ls_trunc1;
    time_ml_trunc1 = ckpt.time_ml_trunc1;
    time_ml_direct = ckpt.time_ml_direct;
    time_ls_est    = ckpt.time_ls_est;
    completed_mask = ckpt.completed_mask;
    fprintf('[Worker %d] Resumed: %d/%d realizations already completed.\n', ...
        part_id, sum(completed_mask), n_sub);
end

t_worker_start = tic;

for i = start_idx:end_idx
    local_idx = i - start_idx + 1;
    if completed_mask(local_idx)
        continue;
    end

    rec = records{i};
    noise_dbm = double(rec.noise_dbm);
    seed_val  = double(rec.seed);
    trial_val = double(rec.trial);

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
    t_ls_e   = double(rec.time_ls_est);

    seeds(local_idx)          = seed_val;
    noise_vec(local_idx)      = noise_dbm;
    trials(local_idx)         = trial_val;
    time_ml_direct(local_idx) = t_ml_inf;
    time_ls_est(local_idx)    = t_ls_e;

    t_real_start = tic;

    %% 1. Perfect-CSI AO (Upper benchmark)
    rng(1);
    [~, P_perf, Theta_perf] = Opt_func_perfectCSI(M, N, K, P_max, sigma, H_true, G_true);
    Phi_perf = diag(conj(Theta_perf));
    res_perfect(local_idx) = compute_sum_rate(H_true' * Phi_perf * G_true, P_perf, K, sigma);

    %% 2. Conventional LS -> Full AO (Main baseline)
    rng(1);
    t0 = tic;
    [~, P_ls_full, Theta_ls_full, it_ls] = Opt_func_Ck(M, N, K, P_max, sigma, C_hat);
    time_ls_full(local_idx) = toc(t0);
    iters_ls_full(local_idx) = it_ls;
    Phi_ls_full = diag(conj(Theta_ls_full));
    res_ls_full(local_idx) = compute_sum_rate(H_true' * Phi_ls_full * G_true, P_ls_full, K, sigma);

    %% 3. Conventional LS -> Truncated AO (1 iteration)
    rng(1);
    t0 = tic;
    [~, P_ls_tr1, Theta_ls_tr1] = Opt_func_Ck(M, N, K, P_max, sigma, C_hat, [], 1);
    time_ls_trunc1(local_idx) = toc(t0);
    Phi_ls_tr1 = diag(conj(Theta_ls_tr1));
    res_ls_trunc1(local_idx) = compute_sum_rate(H_true' * Phi_ls_tr1 * G_true, P_ls_tr1, K, sigma);

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
    res_rand_rzf(local_idx) = compute_sum_rate(H_true' * Phi_rand * G_true, P_rzf, K, sigma);

    %% 5. Proposed Deep Beamforming (Direct ML)
    Phi_ml = diag(conj(theta_ml));
    res_ml_direct(local_idx) = compute_sum_rate(H_true' * Phi_ml * G_true, W_ml, K, sigma);

    %% 6. Proposed ML + Truncated AO (1 iteration refinement)
    t0 = tic;
    [~, P_ml_tr1, Theta_ml_tr1] = Opt_func_Ck(M, N, K, P_max, sigma, C_hat, theta_ml, 1);
    time_ml_trunc1(local_idx) = toc(t0) + t_ml_inf;
    Phi_ml_tr1 = diag(conj(Theta_ml_tr1));
    res_ml_trunc1(local_idx) = compute_sum_rate(H_true' * Phi_ml_tr1 * G_true, P_ml_tr1, K, sigma);

    completed_mask(local_idx) = true;
    t_real = toc(t_real_start);

    gain = res_ml_direct(local_idx) - res_ls_full(local_idx);
    fprintf('[W%d: %3d/%3d] Global #%3d (Seed %6d, %5.1f dBm): LS=%.2f | ML=%.2f | Gain=%+6.2f | Perf=%.2f (%.1fs)\n', ...
        part_id, local_idx, n_sub, i, seed_val, noise_dbm, ...
        res_ls_full(local_idx), res_ml_direct(local_idx), gain, res_perfect(local_idx), t_real);

    % Periodic checkpoint save every 5 realizations
    if mod(sum(completed_mask), 5) == 0 || i == end_idx
        save(ckpt_path, 'seeds', 'noise_vec', 'trials', ...
            'res_perfect', 'res_ls_full', 'res_ls_trunc1', 'res_rand_rzf', ...
            'res_ml_direct', 'res_ml_trunc1', 'time_ls_full', 'iters_ls_full', ...
            'time_ls_trunc1', 'time_ml_trunc1', 'time_ml_direct', 'time_ls_est', ...
            'completed_mask', 'start_idx', 'end_idx', 'part_id');
    end
end

% Final save
save(final_path, 'seeds', 'noise_vec', 'trials', ...
    'res_perfect', 'res_ls_full', 'res_ls_trunc1', 'res_rand_rzf', ...
    'res_ml_direct', 'res_ml_trunc1', 'time_ls_full', 'iters_ls_full', ...
    'time_ls_trunc1', 'time_ml_trunc1', 'time_ml_direct', 'time_ls_est', ...
    'start_idx', 'end_idx', 'part_id');

fprintf('[Worker %d] COMPLETED all %d realizations in %.2f s. Saved to %s\n', ...
    part_id, n_sub, toc(t_worker_start), final_path);
end
