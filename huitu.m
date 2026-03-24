clear all; close all; clc;

fprintf('正在加载仿真数据...\n');
data_lit = load('data_lit.mat');
data_prop = load('data_prop.mat');

test_errs = data_lit.test_errs;
line_colors = lines(length(test_errs)); 
lw = 1.0; 

fig_pos1 = [100, 450, 560, 420]; 
fig_pos2 = [100,  50, 560, 420]; 

figure(1); 
set(gcf, 'Position', fig_pos1);
hold on; box on; grid on;
ylim([-0.8, 0.8]); 

N_lit = length(data_lit.t_span);
plot_idx_lit = 1:50:N_lit;

phi_t_vals = (data_lit.rho_0 - data_lit.rho_inf) * exp(-data_lit.a_ppc * data_lit.t_span) + data_lit.rho_inf;
plot(data_lit.t_span(plot_idx_lit), phi_t_vals(plot_idx_lit), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Upper Bound');
plot(data_lit.t_span(plot_idx_lit), -phi_t_vals(plot_idx_lit), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Lower Bound');

for run_idx = 1:length(test_errs)
    plot(data_lit.t_span(plot_idx_lit), squeeze(data_lit.beta_record(run_idx, 1, plot_idx_lit)), '-', 'Color', line_colors(run_idx,:), ...
        'LineWidth', lw, 'DisplayName', sprintf('Error ($e(0)=%.2f$)', test_errs(run_idx)));
end
xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Tracking Error $z_{1,1}$', 'Interpreter', 'latex');
title('Traditional BLF Control (Singularity Failure for Out-of-Bound Errors)', 'Interpreter', 'latex');
legend('show', 'Location', 'northeastoutside', 'Interpreter', 'latex');

figure(2); 
set(gcf, 'Position', fig_pos2);
hold on; box on; grid on;
ylim([-0.8, 0.8]); 

N_prop = length(data_prop.t_span);
plot_idx_prop = 1:500:N_prop;

U_bound = zeros(1, N_prop);
for k = 1:N_prop
    gamma_t = (1-data_prop.b_f)*exp(-data_prop.a_ppc*data_prop.t_span(k)) + data_prop.b_f;
    if k == 1
        U_bound(k) = NaN; 
    else
        U_bound(k) = gamma_t * sqrt(data_prop.b_s) / sqrt(1 - gamma_t^2);
    end
end

plot(data_prop.t_span(plot_idx_prop), U_bound(plot_idx_prop), 'm--', 'LineWidth', 1.5, 'DisplayName', 'GPPF Upper Bound');
plot(data_prop.t_span(plot_idx_prop), -U_bound(plot_idx_prop), 'b--', 'LineWidth', 1.5, 'DisplayName', 'GPPF Lower Bound');

for run_idx = 1:length(test_errs)
    plot(data_prop.t_span(plot_idx_prop), squeeze(data_prop.beta_record(run_idx, 1, plot_idx_prop)), '-', 'Color', line_colors(run_idx,:), ...
        'LineWidth', lw, 'DisplayName', sprintf('Error ($e(0)=%.2f$)', test_errs(run_idx)));
end
xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Tracking Error $z_{1,1}$', 'Interpreter', 'latex');
title('Proposed GPPF Control (Global Stability Independent of Initial Conditions)', 'Interpreter', 'latex');
legend('show', 'Location', 'northeastoutside', 'Interpreter', 'latex');