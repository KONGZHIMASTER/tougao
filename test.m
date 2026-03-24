clear all; close all; clc; warning off;

dt = 0.001; T = 20; t_span = 0:dt:T; N_steps = length(t_span);
N = 4; n = 2;

psi_i0 = [1, -1, -1, 1];

A_topo = [0, 0, 0, 0; -2.0, 0, 0, 0; 0, 0, 0, 0; 0, 0, -3.0, 0];
B_topo = diag([5, 0, -3, 0]);

kappa_1 = 1.75; kappa_2 = 2.5; beta_zeta = 5; gamma_zeta = 0.75;
A_i = repmat(eye(n), 1, 1, N);

b_s = 0.9975; b_f = 0.05; nu = 1.5;

l = 256; eta_j = 1.0;

W_c = cell(N, 1);
for i = 1:N
    W_c{i} = 2.5 * ones(l, n);
end

Lambda_Q = 1.25;
kappa_3 = 1;

Phi_3 = 25; omega = 0.2; d_1 = [5, 5]; d_2 = [10, 10]; r_switch = 1.25;

iota = [0.5, 0.5]; h_func = [0.75, 0.75];

d_10 = @(t) [sin(2*t)+0.5; sin(2*t)+0.5];
d_21 = @(t) [cos(2*t)+ 0.5*sin(3*t); cos(2*t)+0.5*sin(3*t)];
d_30 = @(t) [sin(0.2*t)+0.2; sin(0.2*t)+0.2];
d_43 = @(t) [cos(2*t);cos(2*t)];
Deception_comm = {d_10, d_21, d_30, d_43};

g = 9.81;
p_i = struct('p1',{},'p2',{},'p3',{},'p4',{},'p5',{});
params_raw = [1.3,1.8,2.0,2.0,1.00,1.00,0.4333,0.6000;
              1.6,1.3,2.1,1.9,1.05,0.95,0.5880,0.3911;
              2.0,1.5,1.8,2.2,0.90,1.10,0.5400,0.6050;
              1.9,1.6,2.1,2.0,1.05,1.00,0.6983,0.5333];
for i = 1:N
    m1=params_raw(i,1); m2=params_raw(i,2); l1=params_raw(i,3); l2=params_raw(i,4);
    r1=params_raw(i,5); r2=params_raw(i,6); J1=params_raw(i,7); J2=params_raw(i,8);
    p_i(i).p1 = m1*r1^2 + m2*(l1^2 + r2^2) + J1 + J2;
    p_i(i).p2 = m2*l1*r2;
    p_i(i).p3 = m2*r2^2 + J2;
    p_i(i).p4 = m1*r1 + m2*l1;
    p_i(i).p5 = m2*r2;
end

Xi_i = repmat(struct('Xi_1',diag([0.5,0.5]),'Xi_2',diag([25,15])),N,1);
phi_bar = [4.0, 3.0, 5.2, 2.7];

J_im = repmat(struct('J_im',[0.5,0.5],'chi_im',[0.75,0.75]),N,1);

C_mat = [0, 1; -1, 0]; 
F_mat = eye(n);

v_hist = zeros(n, N_steps);
q_hist = zeros(n, N_steps, N);
q_dot_hist = zeros(n, N_steps, N);
eta_hist = zeros(n, N_steps, N);
xi_hist = zeros(n, N_steps, N);
d_hat_hist = zeros(N, N+1, N_steps);
Q_hat_hist = zeros(N, N_steps);
vartheta_hist = zeros(n, N_steps, N);
tau_hist = zeros(n, N_steps, N);
e_hist = zeros(n, N_steps, N);
u_hist = zeros(n, N_steps, N);
u_tk_hist = zeros(n, N_steps, N);
F_hat_hist = zeros(n, N_steps, N);
F_true_hist = zeros(n, N_steps, N);
trigger_instants = cell(N, n);

v_hist(:,1) = [3; 0];
q_hist(:,1,:) = [-1.00, -0.75, -0.5, -0.25; -0.5, -0.5, -0.5, -0.5];
eta_hist(:,1,:) = 0 * repmat(eye(n, 1), 1, 1, N);
xi_hist(:,1,:) = zeros(n,1,N);
d_hat_hist(:,:,1) = zeros(N,N+1);
Q_hat_hist(:,1) = [0.25,0.5,0.75,1.0];
vartheta_hist(:,1,:) = 1 * ones(n,1,N);

for i=1:N, e_hist(:,1,i) = q_hist(:,1,i) - F_mat*eta_hist(:,1,i); end

u_tk = zeros(n, N);
zeta = zeros(n, N);

Y = [v_hist(:,1); reshape(q_hist(:,1,:),[],1); zeros(n*N,1);
     reshape(eta_hist(:,1,:),[],1); reshape(xi_hist(:,1,:),[],1);
     reshape(d_hat_hist(:,:,1),[],1); Q_hat_hist(:,1);
     reshape(vartheta_hist(:,1,:),[],1)];

fprintf('Simulation started...\n');
progress_bar_width = 50;
update_interval = 200;

for k = 1:N_steps-1
    [u_tk, zeta, tau_current, trigger_flags, u_continuous] = ...
        calculate_control_and_trigger(Y, t_span(k), u_tk, zeta, N, n, l, ...
        C_mat, F_mat, kappa_1, kappa_2, A_i, beta_zeta, gamma_zeta, b_s, b_f, nu, ...
        eta_j, omega, d_1, d_2, r_switch, Phi_3, p_i, Xi_i, ...
        J_im, g, psi_i0, A_topo, B_topo, Deception_comm, phi_bar, W_c);
    
    u_hist(:,k,:) = u_continuous;
    u_tk_hist(:,k,:) = u_tk;
    tau_hist(:,k,:) = tau_current;
    
    [v_k,q_k,q_dot_k,eta_k,xi_k,~,Q_hat_k,~] = extract_states(Y,N,n,l);
    
    for i = 1:N
        [F_hat_i, F_true_i] = compute_NN_output_and_true_F(i, q_k(:,i), ...
            q_dot_k(:,i), eta_k(:,i), xi_k(:,i), W_c{i}, t_span(k), ...
            eta_j, l, p_i(i), Xi_i(i), g, b_s, b_f, nu, C_mat, F_mat);
        
        F_hat_hist(:,k,i) = F_hat_i;
        F_true_hist(:,k,i) = F_true_i;
    end
    
    for i = 1:N
        for j = 1:n
            if trigger_flags(j, i)
                trigger_instants{i, j} = [trigger_instants{i, j}, t_span(k)];
            end
        end
    end
    
    k1 = calc_derivatives(t_span(k), Y, u_tk, zeta, N, n, l, C_mat, F_mat, ...
        kappa_1, kappa_2, A_i, beta_zeta, gamma_zeta, b_s, b_f, nu, eta_j, ...
        omega, d_1, d_2, r_switch, Phi_3, Lambda_Q, kappa_3, iota, h_func, p_i, ...
        Xi_i, J_im, g, psi_i0, A_topo, B_topo, Deception_comm, W_c);
    k2 = calc_derivatives(t_span(k)+dt/2, Y+dt*k1/2, u_tk, zeta, N, n, l, ...
        C_mat, F_mat, kappa_1, kappa_2, A_i, beta_zeta, gamma_zeta, b_s, b_f, nu, ...
        eta_j, omega, d_1, d_2, r_switch, Phi_3, Lambda_Q, kappa_3, iota, h_func, p_i, ...
        Xi_i, J_im, g, psi_i0, A_topo, B_topo, Deception_comm, W_c);
    k3 = calc_derivatives(t_span(k)+dt/2, Y+dt*k2/2, u_tk, zeta, N, n, l, ...
        C_mat, F_mat, kappa_1, kappa_2, A_i, beta_zeta, gamma_zeta, b_s, b_f, nu, ...
        eta_j, omega, d_1, d_2, r_switch, Phi_3, Lambda_Q, kappa_3, iota, h_func, p_i, ...
        Xi_i, J_im, g, psi_i0, A_topo, B_topo, Deception_comm, W_c);
    k4 = calc_derivatives(t_span(k)+dt, Y+dt*k3, u_tk, zeta, N, n, l, ...
        C_mat, F_mat, kappa_1, kappa_2, A_i, beta_zeta, gamma_zeta, b_s, b_f, nu, ...
        eta_j, omega, d_1, d_2, r_switch, Phi_3, Lambda_Q, kappa_3, iota, h_func, p_i, ...
        Xi_i, J_im, g, psi_i0, A_topo, B_topo, Deception_comm, W_c);
    
    Y = Y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
    
    idx = 1;
    v_hist(:,k+1) = Y(idx:idx+n-1); idx=idx+n;
    q_hist(:,k+1,:) = reshape(Y(idx:idx+n*N-1), [n, N]); idx=idx+n*N;
    q_dot_hist(:,k+1,:) = reshape(Y(idx:idx+n*N-1), [n, N]); idx=idx+n*N;
    eta_hist(:,k+1,:) = reshape(Y(idx:idx+n*N-1), [n, N]); idx=idx+n*N;
    xi_hist(:,k+1,:) = reshape(Y(idx:idx+n*N-1), [n, N]); idx=idx+n*N;
    d_hat_hist(:,:,k+1) = reshape(Y(idx:idx+N*(N+1)-1), [N, N+1]); idx=idx+N*(N+1);
    Q_hat_hist(:,k+1) = Y(idx:idx+N-1); idx=idx+N;
    vartheta_hist(:,k+1,:) = reshape(Y(idx:end), [n, N]);
    
    for i = 1:N, e_hist(:,k+1,i) = q_hist(:,k+1,i) - F_mat*eta_hist(:,k+1,i); end
    
    if mod(k, update_interval) == 0 || k == N_steps-1
        percent_done = k / (N_steps-1);
        filled_length = round(percent_done * progress_bar_width);
        bar_str = ['[' repmat('=', 1, filled_length) repmat(' ', 1, progress_bar_width - filled_length) ']'];
        fprintf('\rSimulation Progress: %s %.1f%%', bar_str, percent_done * 100);
    end
end

fprintf('\nSimulation finished.\n\n');

[v_k,q_k,q_dot_k,eta_k,xi_k,~,Q_hat_k,~] = extract_states(Y,N,n,l);
for i = 1:N
    [F_hat_i, F_true_i] = compute_NN_output_and_true_F(i, q_k(:,i), ...
        q_dot_k(:,i), eta_k(:,i), xi_k(:,i), W_c{i}, t_span(N_steps), ...
        eta_j, l, p_i(i), Xi_i(i), g, b_s, b_f, nu, C_mat, F_mat);
    F_hat_hist(:,N_steps,i) = F_hat_i;
    F_true_hist(:,N_steps,i) = F_true_i;
end

beta_t = (1 - b_f) * exp(-nu * t_span) + b_f;
U_beta = sqrt(b_s )* beta_t ./ sqrt(1 - beta_t.^2);

figure(1);
q_ref_1 = F_mat(1,1)*v_hist(1,:) + F_mat(1,2)*v_hist(2,:);
q_ref_2 = F_mat(2,1)*v_hist(1,:) + F_mat(2,2)*v_hist(2,:);
subplot(4,1,1)
plot(t_span,q_hist(1,:,1),t_span,q_hist(1,:,4),t_span,q_ref_1)
title('Position');
xlabel('Time [sec]');
legend('q_11','q_41','q_01');
subplot(4,1,2)
plot(t_span,q_hist(1,:,2),t_span,q_hist(1,:,3),t_span,-q_ref_1)
title('Position');
xlabel('Time [sec]');
legend('q_21','q_31','-q_01');
subplot(4,1,3)
plot(t_span,q_hist(2,:,1),t_span,q_hist(2,:,4),t_span,q_ref_2)
title('Position');
xlabel('Time [sec]');
legend('q_12','q_42','q_02');
subplot(4,1,4)
plot(t_span,q_hist(2,:,2),t_span,q_hist(2,:,3),t_span,-q_ref_2)
title('Position');
xlabel('Time [sec]');
legend('q_22','q_32','-q_02');

figure(2);
dq_ref_1 = F_mat(1,1)*v_hist(1,:) + F_mat(1,2)*v_hist(2,:);
dq_ref_2 = F_mat(2,1)*v_hist(1,:) + F_mat(2,2)*v_hist(2,:);
subplot(4,1,1)
plot(t_span,q_dot_hist(1,:,1),t_span,q_dot_hist(1,:,4),t_span,dq_ref_2)
title('Velocity');
xlabel('Time [sec]');
legend('dq_11','dq_41','dq_01');
subplot(4,1,2)
plot(t_span,q_dot_hist(1,:,2),t_span,q_dot_hist(1,:,3),t_span,-dq_ref_2)
title('Velocity');
xlabel('Time [sec]');
legend('dq_21','dq_31','-dq_01');
subplot(4,1,3)
plot(t_span,q_dot_hist(2,:,1),t_span,q_dot_hist(2,:,4),t_span,-q_ref_1)
title('Velocity');
xlabel('Time [sec]');
legend('dq_12','dq_42','dq_02');
subplot(4,1,4)
plot(t_span,q_dot_hist(2,:,2),t_span,q_dot_hist(2,:,3), t_span,q_ref_1)
title('Velocity');
xlabel('Time [sec]');
legend('dq_22','dq_32','-dq_02');

figure(3);
dhat_map = [1, 2, 1, 4];
plot(t_span, squeeze(d_hat_hist(1,dhat_map(1),:)));hold on;
legend('d_10');
plot(t_span, squeeze(d_hat_hist(2,dhat_map(2),:)));hold on;
legend('d_21');
plot(t_span, squeeze(d_hat_hist(3,dhat_map(1),:)));hold on;
legend('d_30');
plot(t_span, squeeze(d_hat_hist(4,dhat_map(4),:)));
hold on;
legend('d_43');

figure(4);
subplot(2, 1, 1);
E_Xi = zeros(8, length(t_span));
row = 1;
for dir = 1:2
    for idx = 1:4
        E_Xi(row, :) = squeeze(xi_hist(dir,:,idx));
        row = row + 1;
    end
end
norm2_err_Xi = vecnorm(E_Xi, 2, 1);  
plot(t_span, norm2_err_Xi);
legend('2-norm of auxiliary variable');
xlabel('Time [sec]');
subplot(2, 1, 2);
E = zeros(8, length(t_span));
row = 1;
for dir = 1:2
    for idx = 1:4
        E(row, :) = squeeze(eta_hist(dir,:,idx)) - psi_i0(idx) * v_hist(dir,:);
        row = row + 1;
    end
end
norm2_err = vecnorm(E, 2, 1);  
plot(t_span, norm2_err);
legend('2-norm of observation error');
xlabel('Time [sec]');

figure(5);
subplot(2, 1, 1);
plot(t_span, U_beta, t_span, -U_beta);hold on;
legend('U[beta(t)]','-U[beta(t)]');
plot(t_span, squeeze(e_hist(1,:,1)),t_span, squeeze(e_hist(1,:,2)),t_span, squeeze(e_hist(1,:,3)),t_span, squeeze(e_hist(1,:,4)));
legend('e_11','e_21','e_31','e_41');xlabel('Time [sec]');
subplot(2, 1, 2);
plot(t_span, U_beta, t_span, -U_beta);hold on;
legend('U[beta(t)]','-U[beta(t)]');
plot(t_span, squeeze(e_hist(2,:,1)),t_span, squeeze(e_hist(2,:,2)),t_span, squeeze(e_hist(2,:,3)),t_span, squeeze(e_hist(2,:,4)));
legend('e_12','e_22','e_32','e_42');xlabel('Time [sec]');

figure(6);
for i = 1:N
    subplot(4, 1, i);
    hold on;box on;
    plot(t_span, squeeze(u_hist(1,:,i)));
    plot(t_span, squeeze(u_hist(2,:,i)));
    plot(t_span, squeeze(u_tk_hist(1,:,i)));
    plot(t_span, squeeze(u_tk_hist(2,:,i)));
    legend('\Sigma error norm');
    xlabel('Time [sec]');
end

figure(7);  
hold on;
bin_edges = 0:4:T; 
num_bins = length(bin_edges) - 1;
styles = {'-ob', '-sr', '-dg', '-+k'; '--ob', '--sr', '--dg', '--+k'};
legend_entries = cell(N * n, 1);
plot_handles = zeros(N * n, 1);
plot_idx = 1;

for i = 1:N
    for j = 1:n
        timestamps = trigger_instants{i, j};
        counts_per_bin = histcounts(timestamps, bin_edges);
        cumulative_counts = cumsum(counts_per_bin);
        plot_times_X = bin_edges; 
        plot_counts_Y = [0, cumulative_counts];
        plot_handles(plot_idx) = plot(plot_times_X, plot_counts_Y, ...
            styles{j, i}, 'MarkerSize', 6);
        legend_entries{plot_idx} = sprintf('Follower %d, Direction %d', i, j);
        plot_idx = plot_idx + 1;
    end
end
hold off; grid on; box on;
xlabel('Time (s)'); ylabel('Cumulative Trigger Count');
title('Cumulative Trigger Counts (Starting from t=0)');
legend(plot_handles, legend_entries, 'Location', 'northwest', 'NumColumns', 2);
set(gca, 'XTick', bin_edges); xlim([0, T]);

figure(8);
hold on;
for i = 1:N
    plot(t_span, Q_hat_hist(i,:));
end
grid on; box on;
xlabel('Time (s)');
legend('Q1', 'Q2', 'Q3', 'Q4');

% ========== NEW: Figure 9 - Tracking Error Norms ==========
figure(9);

E_all = zeros(2*N, length(t_span));
for i = 1:N
    E_all(2*i-1, :) = squeeze(e_hist(1,:,i));
    E_all(2*i, :) = squeeze(e_hist(2,:,i));
end
tracking_error_norm = vecnorm(E_all, 2, 1);

subplot(2, 1, 1);
plot(t_span, tracking_error_norm, 'LineWidth', 1.5, 'Color', [0.85, 0.33, 0.10]);
title('Overall Tracking Error Norm (All Agents, All Dimensions)');
xlabel('Time [sec]'); ylabel('||e||_2');
grid on;

subplot(2, 1, 2);
hold on; box on; grid on;
colors = [0, 0.447, 0.741; 0.85, 0.325, 0.098; 0.929, 0.694, 0.125; 0.494, 0.184, 0.556];
for i = 1:N
    E_agent = [squeeze(e_hist(1,:,i)); squeeze(e_hist(2,:,i))];
    norm_agent = vecnorm(E_agent, 2, 1);
    plot(t_span, norm_agent, 'LineWidth', 1.2, 'Color', colors(i,:));
end
legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Location', 'best');
title('Individual Agent Tracking Error Norms');
xlabel('Time [sec]'); ylabel('||e_i||_2');
% ===========================================================

fprintf('\nExporting data to CSV files...\n');

output_folder = 'simulation_data_Original';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

time_data = [t_span'];
csvwrite(fullfile(output_folder, 'time.csv'), time_data);

q_ref_1 = F_mat(1,1)*v_hist(1,:) + F_mat(1,2)*v_hist(2,:);
q_ref_2 = F_mat(2,1)*v_hist(1,:) + F_mat(2,2)*v_hist(2,:);
ref_data = [t_span', q_ref_1', q_ref_2', -q_ref_1', -q_ref_2'];
csvwrite(fullfile(output_folder, 'reference_trajectory.csv'), ref_data);

v_dot_hist = C_mat * v_hist;
dq_ref_1 = F_mat(1,1)*v_dot_hist(1,:) + F_mat(1,2)*v_dot_hist(2,:);
dq_ref_2 = F_mat(2,1)*v_dot_hist(1,:) + F_mat(2,2)*v_dot_hist(2,:);
ref_velocity_data = [t_span', dq_ref_1', dq_ref_2', -dq_ref_1', -dq_ref_2'];
csvwrite(fullfile(output_folder, 'reference_velocity.csv'), ref_velocity_data);

position_data = [t_span', ...
                 squeeze(q_hist(1,:,1))', squeeze(q_hist(2,:,1))', ...
                 squeeze(q_hist(1,:,2))', squeeze(q_hist(2,:,2))', ...
                 squeeze(q_hist(1,:,3))', squeeze(q_hist(2,:,3))', ...
                 squeeze(q_hist(1,:,4))', squeeze(q_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'positions.csv'), position_data);

velocity_data = [t_span', ...
                 squeeze(q_dot_hist(1,:,1))', squeeze(q_dot_hist(2,:,1))', ...
                 squeeze(q_dot_hist(1,:,2))', squeeze(q_dot_hist(2,:,2))', ...
                 squeeze(q_dot_hist(1,:,3))', squeeze(q_dot_hist(2,:,3))', ...
                 squeeze(q_dot_hist(1,:,4))', squeeze(q_dot_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'velocities.csv'), velocity_data);

error_data = [t_span', ...
              squeeze(e_hist(1,:,1))', squeeze(e_hist(2,:,1))', ...
              squeeze(e_hist(1,:,2))', squeeze(e_hist(2,:,2))', ...
              squeeze(e_hist(1,:,3))', squeeze(e_hist(2,:,3))', ...
              squeeze(e_hist(1,:,4))', squeeze(e_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'tracking_errors.csv'), error_data);

torque_data = [t_span', ...
               squeeze(tau_hist(1,:,1))', squeeze(tau_hist(2,:,1))', ...
               squeeze(tau_hist(1,:,2))', squeeze(tau_hist(2,:,2))', ...
               squeeze(tau_hist(1,:,3))', squeeze(tau_hist(2,:,3))', ...
               squeeze(tau_hist(1,:,4))', squeeze(tau_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'control_torques.csv'), torque_data);

beta_t = (1 - b_f) * exp(-nu * t_span) + b_f;
U_beta = sqrt(b_s) * beta_t ./ sqrt(1 - beta_t.^2);
blf_bounds_data = [t_span', U_beta', -U_beta'];
csvwrite(fullfile(output_folder, 'BLF_bounds.csv'), blf_bounds_data);

continuous_control_data = [t_span', ...
                           squeeze(u_hist(1,:,1))', squeeze(u_hist(2,:,1))', ...
                           squeeze(u_hist(1,:,2))', squeeze(u_hist(2,:,2))', ...
                           squeeze(u_hist(1,:,3))', squeeze(u_hist(2,:,3))', ...
                           squeeze(u_hist(1,:,4))', squeeze(u_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'continuous_control.csv'), continuous_control_data);

triggered_control_data = [t_span', ...
                          squeeze(u_tk_hist(1,:,1))', squeeze(u_tk_hist(2,:,1))', ...
                          squeeze(u_tk_hist(1,:,2))', squeeze(u_tk_hist(2,:,2))', ...
                          squeeze(u_tk_hist(1,:,3))', squeeze(u_tk_hist(2,:,3))', ...
                          squeeze(u_tk_hist(1,:,4))', squeeze(u_tk_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'triggered_control.csv'), triggered_control_data);

F_hat_data = [t_span', ...
              squeeze(F_hat_hist(1,:,1))', squeeze(F_hat_hist(2,:,1))', ...
              squeeze(F_hat_hist(1,:,2))', squeeze(F_hat_hist(2,:,2))', ...
              squeeze(F_hat_hist(1,:,3))', squeeze(F_hat_hist(2,:,3))', ...
              squeeze(F_hat_hist(1,:,4))', squeeze(F_hat_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'NN_F_hat.csv'), F_hat_data);

F_true_data = [t_span', ...
               squeeze(F_true_hist(1,:,1))', squeeze(F_true_hist(2,:,1))', ...
               squeeze(F_true_hist(1,:,2))', squeeze(F_true_hist(2,:,2))', ...
               squeeze(F_true_hist(1,:,3))', squeeze(F_true_hist(2,:,3))', ...
               squeeze(F_true_hist(1,:,4))', squeeze(F_true_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'NN_F_true.csv'), F_true_data);

F_error_data = [t_span', ...
                (squeeze(F_hat_hist(1,:,1)) - squeeze(F_true_hist(1,:,1)))', ...
                (squeeze(F_hat_hist(2,:,1)) - squeeze(F_true_hist(2,:,1)))', ...
                (squeeze(F_hat_hist(1,:,2)) - squeeze(F_true_hist(1,:,2)))', ...
                (squeeze(F_hat_hist(2,:,2)) - squeeze(F_true_hist(2,:,2)))', ...
                (squeeze(F_hat_hist(1,:,3)) - squeeze(F_true_hist(1,:,3)))', ...
                (squeeze(F_hat_hist(2,:,3)) - squeeze(F_true_hist(2,:,3)))', ...
                (squeeze(F_hat_hist(1,:,4)) - squeeze(F_true_hist(1,:,4)))', ...
                (squeeze(F_hat_hist(2,:,4)) - squeeze(F_true_hist(2,:,4)))'];
csvwrite(fullfile(output_folder, 'NN_estimation_error.csv'), F_error_data);

Q_hat_data = [t_span', Q_hat_hist(1,:)', Q_hat_hist(2,:)', ...
              Q_hat_hist(3,:)', Q_hat_hist(4,:)'];
csvwrite(fullfile(output_folder, 'Q_hat.csv'), Q_hat_data);

E = zeros(8, length(t_span));
row = 1;
for dir = 1:2
    for idx = 1:4
        E(row, :) = squeeze(eta_hist(dir,:,idx)) - psi_i0(idx) * v_hist(dir,:);
        row = row + 1;
    end
end
norm2_err = vecnorm(E, 2, 1);
observer_error_data = [t_span', norm2_err'];
csvwrite(fullfile(output_folder, 'observer_error_norm.csv'), observer_error_data);

E_Xi = zeros(8, length(t_span));
row = 1;
for dir = 1:2
    for idx = 1:4
        E_Xi(row, :) = squeeze(xi_hist(dir,:,idx));
        row = row + 1;
    end
end
norm2_err_Xi = vecnorm(E_Xi, 2, 1);
xi_norm_data = [t_span', norm2_err_Xi'];
csvwrite(fullfile(output_folder, 'xi_norm.csv'), xi_norm_data);

dhat_map = [1, 2, 1, 4];
d_hat_data = [t_span', ...
              squeeze(d_hat_hist(1,dhat_map(1),:))', ...
              squeeze(d_hat_hist(2,dhat_map(2),:))', ...
              squeeze(d_hat_hist(3,dhat_map(1),:))', ...
              squeeze(d_hat_hist(4,dhat_map(4),:))'];
csvwrite(fullfile(output_folder, 'd_hat_estimates.csv'), d_hat_data);

bin_edges = 0:4:T;
num_bins = length(bin_edges) - 1;
trigger_counts = zeros(N*n, num_bins);
trigger_cumulative = zeros(N*n, length(bin_edges));

for i = 1:N
    for j = 1:n
        idx = (i-1)*n + j;
        timestamps = trigger_instants{i, j};
        counts_per_bin = histcounts(timestamps, bin_edges);
        trigger_counts(idx, :) = counts_per_bin;
        trigger_cumulative(idx, :) = [0, cumsum(counts_per_bin)];
    end
end

trigger_count_data = [bin_edges(1:end-1)', trigger_counts'];
csvwrite(fullfile(output_folder, 'trigger_counts_per_bin.csv'), trigger_count_data);

trigger_cumulative_data = [bin_edges', trigger_cumulative'];
csvwrite(fullfile(output_folder, 'trigger_cumulative.csv'), trigger_cumulative_data);

vartheta_data = [t_span', ...
                 squeeze(vartheta_hist(1,:,1))', squeeze(vartheta_hist(2,:,1))', ...
                 squeeze(vartheta_hist(1,:,2))', squeeze(vartheta_hist(2,:,2))', ...
                 squeeze(vartheta_hist(1,:,3))', squeeze(vartheta_hist(2,:,3))', ...
                 squeeze(vartheta_hist(1,:,4))', squeeze(vartheta_hist(2,:,4))'];
csvwrite(fullfile(output_folder, 'vartheta.csv'), vartheta_data);

% ========== NEW: Export Tracking Error Norms ==========
tracking_error_norm_data = [t_span', tracking_error_norm'];
csvwrite(fullfile(output_folder, 'tracking_error_norm_overall.csv'), tracking_error_norm_data);

individual_norms = zeros(length(t_span), N);
for i = 1:N
    E_agent = [squeeze(e_hist(1,:,i)); squeeze(e_hist(2,:,i))];
    individual_norms(:,i) = vecnorm(E_agent, 2, 1)';
end
tracking_error_norm_individual = [t_span', individual_norms];
csvwrite(fullfile(output_folder, 'tracking_error_norm_individual.csv'), tracking_error_norm_individual);
% ======================================================

position_table = array2table(position_data, ...
    'VariableNames', {'Time', 'q11', 'q12', 'q21', 'q22', 'q31', 'q32', 'q41', 'q42'});
writetable(position_table, fullfile(output_folder, 'positions_with_header.csv'));

error_table = array2table(error_data, ...
    'VariableNames', {'Time', 'e11', 'e12', 'e21', 'e22', 'e31', 'e32', 'e41', 'e42'});
writetable(error_table, fullfile(output_folder, 'errors_with_header.csv'));

torque_table = array2table(torque_data, ...
    'VariableNames', {'Time', 'tau11', 'tau12', 'tau21', 'tau22', 'tau31', 'tau32', 'tau41', 'tau42'});
writetable(torque_table, fullfile(output_folder, 'torques_with_header.csv'));

blf_table = array2table(blf_bounds_data, ...
    'VariableNames', {'Time', 'U_beta', 'minus_U_beta'});
writetable(blf_table, fullfile(output_folder, 'BLF_bounds_with_header.csv'));

continuous_table = array2table(continuous_control_data, ...
    'VariableNames', {'Time', 'u11', 'u12', 'u21', 'u22', 'u31', 'u32', 'u41', 'u42'});
writetable(continuous_table, fullfile(output_folder, 'continuous_control_with_header.csv'));

triggered_table = array2table(triggered_control_data, ...
    'VariableNames', {'Time', 'u_tk11', 'u_tk12', 'u_tk21', 'u_tk22', 'u_tk31', 'u_tk32', 'u_tk41', 'u_tk42'});
writetable(triggered_table, fullfile(output_folder, 'triggered_control_with_header.csv'));

F_hat_table = array2table(F_hat_data, ...
    'VariableNames', {'Time', 'F_hat11', 'F_hat12', 'F_hat21', 'F_hat22', 'F_hat31', 'F_hat32', 'F_hat41', 'F_hat42'});
writetable(F_hat_table, fullfile(output_folder, 'F_hat_with_header.csv'));

F_true_table = array2table(F_true_data, ...
    'VariableNames', {'Time', 'F_true11', 'F_true12', 'F_true21', 'F_true22', 'F_true31', 'F_true32', 'F_true41', 'F_true42'});
writetable(F_true_table, fullfile(output_folder, 'F_true_with_header.csv'));

F_error_table = array2table(F_error_data, ...
    'VariableNames', {'Time', 'error11', 'error12', 'error21', 'error22', 'error31', 'error32', 'error41', 'error42'});
writetable(F_error_table, fullfile(output_folder, 'NN_error_with_header.csv'));

Q_hat_table = array2table(Q_hat_data, ...
    'VariableNames', {'Time', 'Q_hat1', 'Q_hat2', 'Q_hat3', 'Q_hat4'});
writetable(Q_hat_table, fullfile(output_folder, 'Q_hat_with_header.csv'));

ref_table = array2table(ref_data, ...
    'VariableNames', {'Time', 'q_ref1', 'q_ref2', 'minus_q_ref1', 'minus_q_ref2'});
writetable(ref_table, fullfile(output_folder, 'reference_with_header.csv'));

ref_velocity_table = array2table(ref_velocity_data, ...
    'VariableNames', {'Time', 'dq_ref1', 'dq_ref2', 'minus_dq_ref1', 'minus_dq_ref2'});
writetable(ref_velocity_table, fullfile(output_folder, 'reference_velocity_with_header.csv'));

d_hat_table = array2table(d_hat_data, ...
    'VariableNames', {'Time', 'd_hat10', 'd_hat21', 'd_hat30', 'd_hat43'});
writetable(d_hat_table, fullfile(output_folder, 'd_hat_with_header.csv'));

trigger_names = cell(1, N*n+1);
trigger_names{1} = 'Time';
for i = 1:N
    for j = 1:n
        idx = (i-1)*n + j + 1;
        trigger_names{idx} = sprintf('Agent%d_Dir%d', i, j);
    end
end
trigger_cumulative_table = array2table(trigger_cumulative_data, 'VariableNames', trigger_names);
writetable(trigger_cumulative_table, fullfile(output_folder, 'trigger_cumulative_with_header.csv'));

vartheta_table = array2table(vartheta_data, ...
    'VariableNames', {'Time', 'vartheta11', 'vartheta12', 'vartheta21', 'vartheta22', ...
                      'vartheta31', 'vartheta32', 'vartheta41', 'vartheta42'});
writetable(vartheta_table, fullfile(output_folder, 'vartheta_with_header.csv'));

% ========== NEW: Export with headers ==========
overall_norm_table = array2table(tracking_error_norm_data, ...
    'VariableNames', {'Time', 'Overall_Tracking_Error_Norm'});
writetable(overall_norm_table, fullfile(output_folder, 'tracking_error_norm_overall_with_header.csv'));

individual_norm_table = array2table(tracking_error_norm_individual, ...
    'VariableNames', {'Time', 'Agent1_Norm', 'Agent2_Norm', 'Agent3_Norm', 'Agent4_Norm'});
writetable(individual_norm_table, fullfile(output_folder, 'tracking_error_norm_individual_with_header.csv'));
% =============================================

fprintf('CSV export completed! Files saved in "%s" folder.\n', output_folder);

function [u_tk_new, zeta_new, tau, trigger_flags, u] = ...
    calculate_control_and_trigger(Y, t, u_tk_old, zeta_old, N, n, l, ...
    C_mat, F_mat, kappa_1, kappa_2, A_i, beta_zeta, gamma_zeta, b_s, b_f, nu, ...
    eta_j, omega, d_1, d_2, r_switch, Phi_3, p_i, Xi_i, ...
    J_im, g, psi_i0, A_topo, B_topo, Deception_comm, phi_bar, W_c)
    
    [v,q,q_dot,eta,xi,d_hat,Q_hat,vartheta] = extract_states(Y,N,n,l);
    
    u = zeros(n, N); tau = zeros(n, N);
    u_tk_new = u_tk_old; zeta_new = zeta_old;
    trigger_flags = zeros(n, N);
    
    eta_dot = C_mat * eta + xi;
    xi_dot = zeros(n, N);
    h_t = beta_zeta * exp(-gamma_zeta * t);
    
    xi_dot(:,1) = update_observer(1, xi, eta, v, d_hat, A_i, psi_i0, B_topo, A_topo, ...
        kappa_1, h_t, F_mat, Deception_comm{1}(t), []);
    xi_dot(:,2) = update_observer(2, xi, eta, [], d_hat, A_i, psi_i0, B_topo, A_topo, ...
        kappa_1, h_t, F_mat, Deception_comm{2}(t), eta(:,1));
    xi_dot(:,3) = update_observer(3, xi, eta, v, d_hat, A_i, psi_i0, B_topo, A_topo, ...
        kappa_1, h_t, F_mat, Deception_comm{3}(t), []);
    xi_dot(:,4) = update_observer(4, xi, eta, [], d_hat, A_i, psi_i0, B_topo, A_topo, ...
        kappa_1, h_t, F_mat, Deception_comm{4}(t), eta(:,3));
    
    eta_ddot = C_mat * eta_dot + xi_dot;
    
    for i = 1:N
        [u(:,i), tau(:,i)] = compute_control(i, q(:,i), q_dot(:,i), eta(:,i), ...
            eta_dot(:,i), eta_ddot(:,i), W_c{i}, Q_hat(i), t, b_s, b_f, nu, eta_j, ...
            omega, d_1, d_2, r_switch, l, p_i(i), Xi_i(i), ...
            g, phi_bar(i), u_tk_new(:,i), F_mat);
        
        zeta_new(:,i) = u(:,i) - u_tk_new(:,i);
        
        z_i = q_dot(:,i) - compute_alpha(q(:,i), q_dot(:,i), eta(:,i), ...
            eta_dot(:,i), t, b_s, b_f, nu, Xi_i(i), F_mat);
        for j = 1:n
            if J_im(i).J_im(j)*(zeta_new(j,i)^2 - ...
                    J_im(i).chi_im(j)*z_i(j)^2) >= vartheta(j,i)
                u_tk_new(j,i) = u(j,i);
                zeta_new(j,i) = 0;
                trigger_flags(j,i) = 1;
            end
        end
    end
end

function xi_dot_i = update_observer(i, xi, eta, v, d_hat, A_i, psi_i0, B_topo, A_topo, ...
    kappa_1, h_t, F_mat, deception, neighbor_eta)
    
    norm_xiP = norm(xi(:,i)' * A_i(:,:,i));
    d_hat_term = -(A_i(:,:,i)' * xi(:,i) * d_hat(i,get_dhat_idx(i))) / (norm_xiP + h_t);
    
    if isempty(neighbor_eta)
        ref = F_mat*v + deception;
        weight = B_topo(i,i);
        xi_dot_i = -xi(:,i) + kappa_1*psi_i0(i)*weight*(psi_i0(i)*ref - F_mat*eta(:,i)) + ...
            psi_i0(i)*kappa_1*weight*d_hat_term;
    else
        ref = F_mat*neighbor_eta + deception;
        weight = A_topo(i,get_neighbor_idx(i));
        if i == 4
            xi_dot_i = -xi(:,i) + kappa_1*psi_i0(3)*weight*(psi_i0(3)*ref - F_mat*eta(:,i)) + ...
                psi_i0(3)*kappa_1*weight*d_hat_term;
        else
            xi_dot_i = -xi(:,i) + kappa_1*psi_i0(i)*weight*(psi_i0(i)*ref - F_mat*eta(:,i)) + ...
                psi_i0(i)*kappa_1*weight*d_hat_term;
        end
    end
end

function [u_i, tau_i] = compute_control(i, q_i, q_dot_i, eta_i, eta_dot_i, eta_ddot_i, ...
    W_c_i, Q_hat_i, t, b_s, b_f, nu, eta_j, omega, d_1, d_2, r_switch, l, p_i, ...
    Xi_i, g, phi_bar_i, u_tk_i, F_mat)
    
    alpha_i = compute_alpha(q_i, q_dot_i, eta_i, eta_dot_i, t, b_s, b_f, nu, Xi_i, F_mat);
    alpha_dot_i = compute_alpha(q_i, q_dot_i, eta_i, eta_dot_i, t+1e-6, b_s, b_f, nu, Xi_i, F_mat);
    alpha_dot_i = (alpha_dot_i - alpha_i) / 1e-6;
    
    z_i = q_dot_i - alpha_i;
    Z_i = [q_i; q_dot_i; alpha_i; alpha_dot_i];
    
    rbf_centers = linspace(-1, 1, l);
    S_i = exp(-sum((Z_i - rbf_centers).^2, 1)' / (2*eta_j^2));
    
    Pi_i = compute_switching(Z_i, d_1, d_2, r_switch);
    
    M_i = [p_i.p1+2*p_i.p2*cos(q_i(2)), p_i.p3+p_i.p2*cos(q_i(2));
           p_i.p3+p_i.p2*cos(q_i(2)), p_i.p3];
    C_i = [-p_i.p2*q_dot_i(2)*sin(q_i(2)), -p_i.p2*(q_dot_i(1)+q_dot_i(2))*sin(q_i(2));
           p_i.p2*q_dot_i(1)*sin(q_i(2)), 0];
    G_i = [p_i.p4*g*cos(q_i(1))+p_i.p5*g*cos(q_i(1)+q_i(2));
           p_i.p5*g*cos(q_i(1)+q_i(2))];
    
    [~, tau_d_i2] = get_tau_attacks(i, q_i, q_dot_i, t);
    F_U_vec = M_i*alpha_dot_i + C_i*alpha_i + G_i - tau_d_i2;
    F_U_i = [norm(F_U_vec)/sqrt(2); norm(F_U_vec)/sqrt(2)];
    
    Theta_i = F_U_i .* tanh(F_U_i .* z_i / omega);
    
    W_c_norm = norm(W_c_i, 'fro');
    S_i_norm = norm(S_i);
    
    sgn_z_i = sat(z_i / omega);
    
    NN_term = zeros(2,1);
    for j = 1:2
        NN_term(j) = Pi_i(j,j) * (Q_hat_i + W_c_norm) * S_i_norm * sgn_z_i(j);
    end
    
    u_i2 = -(Pi_i * NN_term + (eye(2) - Pi_i) * Theta_i);
    u_i = -Xi_i.Xi_2 * z_i + u_i2 / phi_bar_i;
    
    phi_i = get_phi_attack(i, t);
    [tau_d_i1, ~] = get_tau_attacks(i, q_i, q_dot_i, t);
    tau_i = phi_i*u_tk_i + tau_d_i1 + tau_d_i2;
end

function alpha = compute_alpha(q_i, q_dot_i, eta_i, eta_dot_i, t, b_s, b_f, ...
    nu, Xi_i, F_mat)
    e_i = q_i - F_mat*eta_i;
    theta_inv = 1/((1-b_f)*exp(-nu*t) + b_f);
    gamma_i = e_i ./ sqrt(e_i.^2 + b_s);
    theta_i = theta_inv * gamma_i;
    varphi_i = (1 + theta_i.^2) ./ (1 - theta_i.^2).^2;
    sigma_i = sqrt(b_s) ./ ((e_i.^2 + b_s).^(3/4));
    X_i1_diag = varphi_i .* sigma_i .* theta_inv;
    beta_dot = -nu*(1-b_f)*exp(-nu*t);
    theta_inv_dot = -beta_dot/((1-b_f)*exp(-nu*t) + b_f)^2;
    X_i2 = varphi_i .* theta_inv_dot .* gamma_i;
    rho_i = theta_i ./ (1 - theta_i.^2);
    alpha = -X_i2./X_i1_diag - Xi_i.Xi_1*diag(X_i1_diag)*rho_i + F_mat*eta_dot_i;
end

function Pi_i = compute_switching(Z, d_1, d_2, r_switch)
    Pi_i = zeros(2,2);
    for j = 1:2
        Z_norm = norm(Z(2*j-1:2*j));
        if Z_norm < d_1(j), m_func = 1;
        elseif Z_norm > d_2(j), m_func = 0;
        else
            exp_arg = (Z_norm^2-d_1(j)^2)/(r_switch*(d_2(j)^2-d_1(j)^2));
            m_func = (d_2(j)^2-Z_norm^2)/(d_2(j)^2-d_1(j)^2)*exp(-exp_arg^2);
        end
        Pi_i(j,j) = m_func;
    end
end

function phi_i = get_phi_attack(i, t)
    switch i
        case 1, phi_i = 4 + 6*exp(-0.1*t);
        case 2, phi_i = 3.0 + 4.4*abs(sin(t/5));
        case 3, phi_i = 4.2 + min(exp(0.15*t),exp(2));
        case 4, phi_i = 2.7 + 6.3*t/(1 + 0.1*t^2);
    end
end

function [tau_d_i1, tau_d_i2] = get_tau_attacks(i, q_i, q_dot_i, t)
    if i==1 || i==2
        tau_d_i1 = [0.6*exp(-0.08*t)+0.4*sin(0.5*t); 0.5*exp(-0.06*t)+0.3*cos(0.3*t)];
    else
        tau_d_i1 = [0.25*q_dot_i(1)/(1+q_dot_i(1)^2)+0.5*t/(1+0.2*t^2);
                  0.35*q_dot_i(2)/(1+q_dot_i(2)^2)+0.4*t/(1+0.15*t^2)];
    end
    if i==1 || i==3
        tau_d_i2 = [1.8*tanh(q_i(1)); 1.3*cos(1.5*q_i(2))];
    else
        tau_d_i2 = [1.3*tanh(q_i(1)+q_dot_i(1)); 0.7*cos(q_i(2)*q_dot_i(2))];
    end
end

function idx = get_dhat_idx(i)
    map = [1, 2, 1, 4]; idx = map(i);
end

function idx = get_neighbor_idx(i)
    map = [0, 1, 0, 3]; idx = map(i);
end

function dY = calc_derivatives(t, Y, u_tk, zeta, N, n, l, C_mat, F_mat, ...
    kappa_1, kappa_2, A_i, beta_zeta, gamma_zeta, b_s, b_f, nu, eta_j, ...
    omega, d_1, d_2, r_switch, Phi_3, Lambda_Q, kappa_3, iota, h_func, p_i, Xi_i, ...
    J_im, g, psi_i0, A_topo, B_topo, Deception_comm, W_c)
    
    [v,q,q_dot,eta,xi,d_hat,Q_hat,vartheta] = extract_states(Y,N,n,l);
    
    v_dot = C_mat * v;
    q_ddot = zeros(n, N);
    eta_dot = C_mat * eta + xi;
    xi_dot = zeros(n, N);
    d_hat_dot = zeros(N, N+1);
    Q_hat_dot = zeros(N, 1);
    vartheta_dot = zeros(n, N);
    
    h_t = beta_zeta * exp(-gamma_zeta * t);
    
    for i = 1:N
        norm_xi_P = norm(xi(:,i)' * A_i(:,:,i));
        d_hat_dot(i,get_dhat_idx(i)) = kappa_2 * (norm_xi_P - h_t*d_hat(i,get_dhat_idx(i)));
    end
    
    xi_dot(:,1) = update_observer(1, xi, eta, v, d_hat, A_i, psi_i0, B_topo, A_topo, ...
        kappa_1, h_t, F_mat, Deception_comm{1}(t), []);
    xi_dot(:,2) = update_observer(2, xi, eta, [], d_hat, A_i, psi_i0, B_topo, A_topo, ...
        kappa_1, h_t, F_mat, Deception_comm{2}(t), eta(:,1));
    xi_dot(:,3) = update_observer(3, xi, eta, v, d_hat, A_i, psi_i0, B_topo, A_topo, ...
        kappa_1, h_t, F_mat, Deception_comm{3}(t), []);
    xi_dot(:,4) = update_observer(4, xi, eta, [], d_hat, A_i, psi_i0, B_topo, A_topo, ...
        kappa_1, h_t, F_mat, Deception_comm{4}(t), eta(:,3));
    
    eta_ddot = C_mat * eta_dot + xi_dot;
    
    for i = 1:N
        phi_i = get_phi_attack(i, t);
        [tau_d_i1, tau_d_i2] = get_tau_attacks(i, q(:,i), q_dot(:,i), t);
        tau_i = phi_i*u_tk(:,i) + tau_d_i1 + tau_d_i2;
        
        p = p_i(i);
        M_i = [p.p1+2*p.p2*cos(q(2,i)), p.p3+p.p2*cos(q(2,i));
               p.p3+p.p2*cos(q(2,i)), p.p3];
        C_i = [-p.p2*q_dot(2,i)*sin(q(2,i)), -p.p2*(q_dot(1,i)+q_dot(2,i))*sin(q(2,i));
               p.p2*q_dot(1,i)*sin(q(2,i)), 0];
        G_i = [p.p4*g*cos(q(1,i))+p.p5*g*cos(q(1,i)+q(2,i));
               p.p5*g*cos(q(1,i)+q(2,i))];
        
        q_ddot(:,i) = M_i\(tau_i - C_i*q_dot(:,i) - G_i);
        
        alpha_i = compute_alpha(q(:,i), q_dot(:,i), eta(:,i), eta_dot(:,i), ...
            t, b_s, b_f, nu, Xi_i(i), F_mat);
        z_i = q_dot(:,i) - alpha_i;
        Z_i = [q(:,i); q_dot(:,i); alpha_i; compute_alpha(q(:,i), q_dot(:,i), ...
            eta(:,i), eta_dot(:,i), t+1e-6, b_s, b_f, nu, Xi_i(i), F_mat)];
        
        rbf_centers = linspace(-1, 1, l);
        S_i = exp(-sum((Z_i - rbf_centers).^2, 1)' / (2*eta_j^2));
        Pi_i = compute_switching(Z_i, d_1, d_2, r_switch);
        
        pi_i_scalar = Pi_i(1,1) * Pi_i(2,2);
        z_iS_i_norm = norm(z_i*S_i');
        Q_hat_dot(i) = Lambda_Q * (pi_i_scalar * z_iS_i_norm - kappa_3 * Q_hat(i));
        
        vartheta_dot(:,i) = -iota'.*vartheta(:,i) + h_func'.*(J_im(i).chi_im'.*z_i.^2 - zeta(:,i).^2);
    end
    
    dY = [v_dot; reshape(q_dot,[],1); reshape(q_ddot,[],1); reshape(eta_dot,[],1);
          reshape(xi_dot,[],1); reshape(d_hat_dot,[],1); Q_hat_dot;
          reshape(vartheta_dot,[],1)];
end

function [v,q,q_dot,eta,xi,d_hat,Q_hat,vartheta] = extract_states(Y,N,n,l)
    idx = 1;
    v = Y(idx:idx+n-1); idx=idx+n;
    q = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    q_dot = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    eta = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    xi = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    d_hat = reshape(Y(idx:idx+N*(N+1)-1), [N,N+1]); idx=idx+N*(N+1);
    Q_hat = Y(idx:idx+N-1); idx=idx+N;
    vartheta = reshape(Y(idx:end), [n,N]);
end

function [F_hat_i, F_true_i] = compute_NN_output_and_true_F(i, q_i, q_dot_i, ...
    eta_i, xi_i, W_c_i, t, eta_j, l, p_i, Xi_i, g, b_s, ...
    b_f, nu, C_mat, F_mat)
    
    eta_dot_i = C_mat * eta_i + xi_i;
    eta_ddot_i = C_mat * eta_dot_i;
    
    e_i = q_i - F_mat*eta_i;
    e_dot_i = q_dot_i - F_mat*eta_dot_i;
    
    beta = (1-b_f)*exp(-nu*t) + b_f;
    beta_dot = -nu*(1-b_f)*exp(-nu*t);
    theta_inv = 1/beta;
    theta_inv_dot = -beta_dot/beta^2;
    
    gamma_i = e_i ./ sqrt(e_i.^2 + b_s);
    gamma_i_dot = (b_s*e_dot_i) ./ ((e_i.^2 + b_s).^(3/2));
    
    theta_i = theta_inv * gamma_i;
    theta_dot_i = theta_inv_dot*gamma_i + theta_inv*gamma_i_dot;
    
    rho_i = theta_i ./ (1 - theta_i.^2);
    rho_dot_i = (theta_dot_i .* (1 + theta_i.^2)) ./ ((1 - theta_i.^2).^2);
    
    varphi_i = (1 + theta_i.^2) ./ (1 - theta_i.^2).^2;
    sigma_i = sqrt(b_s) ./ ((e_i.^2 + b_s).^(3/4));
    X_i1_diag = varphi_i .* sigma_i .* theta_inv;
    X_i2 = varphi_i .* theta_inv_dot .* gamma_i;
    
    alpha_i = -X_i2./X_i1_diag - Xi_i.Xi_1*diag(X_i1_diag)*rho_i + F_mat*eta_dot_i;
    alpha_dot_i = -Xi_i.Xi_1*diag(X_i1_diag)*rho_dot_i + F_mat*eta_ddot_i;
    
    Z_i = [q_i; q_dot_i; alpha_i; alpha_dot_i];
    
    rbf_centers = linspace(-1, 1, l);
    S_i = exp(-sum((Z_i - rbf_centers).^2, 1)' / (2*eta_j^2));
    
    F_hat_i = W_c_i' * S_i;
    
    M_i = [p_i.p1+2*p_i.p2*cos(q_i(2)), p_i.p3+p_i.p2*cos(q_i(2));
           p_i.p3+p_i.p2*cos(q_i(2)), p_i.p3];
    
    C_i = [-p_i.p2*q_dot_i(2)*sin(q_i(2)), -p_i.p2*(q_dot_i(1)+q_dot_i(2))*sin(q_i(2));
           p_i.p2*q_dot_i(1)*sin(q_i(2)), 0];
    
    G_i = [p_i.p4*g*cos(q_i(1)) + p_i.p5*g*cos(q_i(1)+q_i(2));
           p_i.p5*g*cos(q_i(1)+q_i(2))];
    
    [~, tau_d_i2] = get_tau_attacks(i, q_i, q_dot_i, t);
    
    F_true_i = -(M_i*alpha_dot_i + C_i*alpha_i + G_i - tau_d_i2);
end

function y = sat(x)
    y = x;
    y(x >  1) =  1;
    y(x < -1) = -1;
end