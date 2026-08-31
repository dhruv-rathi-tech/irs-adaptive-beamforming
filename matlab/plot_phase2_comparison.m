% plot_phase2_comparison.m
% Bar chart comparing Perfect-CSI AO vs Estimated-CSI AO (algorithm-seen vs
% true evaluation), for ONE channel realization at ONE noise floor.
% This is a single-point illustrative comparison, NOT a Phase 4 sweep --
% label it honestly when presenting.
%
% Run after run_phase2.m (needs results/phase2_results.mat)

load('results/phase2_results.mat');

% use run1 (good SNR case) -- change to 'run2' for the mid-SNR case
r = results.run1;

vals = [r.SumRate_perfect, r.SumRate_estCk_algSeen, r.SumRate_estCk_trueEval];
labels = {'Perfect-CSI AO', 'Estimated-CSI AO\n(algorithm-seen)', 'Estimated-CSI AO\n(TRUE, realistic)'};

figure('Position', [100 100 700 500]);
b = bar(vals, 'FaceColor', 'flat');
b.CData(1,:) = [0.2 0.6 0.2];   % green: upper bound
b.CData(2,:) = [0.9 0.6 0.1];   % orange: optimistic self-view
b.CData(3,:) = [0.8 0.2 0.2];   % red: realistic performance

set(gca, 'XTickLabel', labels);
ylabel('Sum Rate (bps/Hz)');
title({'Phase 2: Classical Baseline Comparison (single realization)', ...
       sprintf('Pilot overhead: %d symbols | C\\_hat mean error: %.1f%%', ...
               r.tau_total, 100*mean(r.recovery_err))}, 'FontSize', 11);
grid on;

for i = 1:length(vals)
    text(i, vals(i)+0.5, sprintf('%.2f', vals(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

saveas(gcf, 'results/fig3_phase2_comparison.png');
fprintf('Saved results/fig3_phase2_comparison.png\n');
