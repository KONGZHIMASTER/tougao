clear all; close all; clc; warning off;

%% --- 系统参数与初始化 ---
dt = 0.0001; T = 20; t_span = 0:dt:T; N_steps = length(t_span);
N = 1; 
n = 2;

psi0 = [1];

scrS0 = [0, 1; -1, 0]; 

rho_0 = 0.5;  
rho_inf = 0.02; 
a_ppc = 1.0;
k_a = 0.1; 

l_nn = 256;
rbf_width = 1.0;
Lambda_W = 1.00; 
gamma_W = 0.5;   
varpi = 0.2; 
d_switch_1 = [2, 2]; 
d_switch_2 = [4, 4]; 
r_switch = 1.25;

g_grav = 9.81;

p_i = struct('p1',{},'p2',{},'p3',{},'p4',{},'p5',{});
params_raw = [1.3,1.8,2.0,2.0,1.00,1.00,0.4333,0.6000]; 
for i = 1:N
    m1=params_raw(i,1); m2=params_raw(i,2); l1=params_raw(i,3); l2=params_raw(i,4);
    r1=params_raw(i,5); r2=params_raw(i,6); J1=params_raw(i,7); J2=params_raw(i,8);
    p_i(i).p1 = m1*r1^2 + m2*(l1^2 + r2^2) + J1 + J2;
    p_i(i).p2 = m2*l1*r2;
    p_i(i).p3 = m2*r2^2 + J2;
    p_i(i).p4 = m1*r1 + m2*l1;
    p_i(i).p5 = m2*r2;
end


K_gain = repmat(struct('K1',diag([2.5,2.5]),'K2',diag([75,75])),N,1);
Phi_min = [1]; 

fprintf('单智能体仿真开始 (文献方法验证, 初始位置 q1=0)...\n');
update_interval = 2000;

v_hist = zeros(n, N_steps);
q_hist = zeros(n, N_steps, N);
q_dot_hist = zeros(n, N_steps, N);
W_hat_hist = zeros(l_nn, n, N_steps);
u_hist = zeros(n, N_steps, N);
alpha_hist = zeros(n, N_steps, N); 

v_hist(:,1) = [0.2; 0.1]; 

q_hist(:,1,1) = [0.0; 0.0]; 
q_dot_hist(:,1,1) = [0.0; 0.0];
W_hat_hist(:,:,1) = 0.01 * ones(l_nn, n);

Y = [v_hist(:,1);
     reshape(q_hist(:,1,:),[],1);
     reshape(q_dot_hist(:,1,:),[],1);
     reshape(W_hat_hist(:,:,1), [], 1)];

%% --- 2. 仿真主循环 ---
for k = 1:N_steps-1
    t_curr = t_span(k);
    
    [u_cont, alpha_cont] = calculate_control(Y, t_curr, N, n, l_nn, ...
        rho_0, rho_inf, a_ppc, k_a, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, p_i, K_gain, ...
        g_grav, Phi_min, psi0, scrS0);
    
    u_hist(:,k,:) = u_cont;
    alpha_hist(:,k,:) = alpha_cont;
    
    dY = derivatives_combined(t_curr, Y, u_cont, N, n, l_nn, scrS0, ...
          psi0, rho_0, rho_inf, a_ppc, k_a, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, Lambda_W, gamma_W, ...
          p_i, K_gain, g_grav);
          
    Y = Y + dt * dY;
    
    [v_new, q_new, q_dot_new, W_hat_new] = extract_all_states(Y, N, n, l_nn);
    
    v_hist(:,k+1) = v_new;
    q_hist(:,k+1,:) = q_new;
    q_dot_hist(:,k+1,:) = q_dot_new;
    W_hat_hist(:,:,k+1) = W_hat_new;
    
    if mod(k, update_interval) == 0 
        fprintf('当前进度: %.1f%%\n', k / (N_steps-1) * 100);
    end
end
u_hist(:,N_steps,:) = u_hist(:,N_steps-1,:);
alpha_hist(:,N_steps,:) = alpha_hist(:,N_steps-1,:);

fprintf('仿真完成，开始绘图并导出数据...\n');

%% --- 3. 绘图部分 ---

v1_ref = psi0(1) * squeeze(v_hist(1,:));
q1_act = squeeze(q_hist(1,:,1));
z1_err = q1_act - v1_ref;

plot_idx = 1:500:N_steps;
lw = 1.5; 
figure(1); set(gcf, 'Position', [100, 100, 600, 450]);
hold on; box on;

plot(t_span(plot_idx), q1_act(plot_idx), 'b-', 'LineWidth', lw, 'DisplayName', 'Actual Trajectory $q_{1,1}$');
plot(t_span(plot_idx), v1_ref(plot_idx), 'r--', 'LineWidth', lw, 'DisplayName', 'Leader Trajectory $v_{1}$');

xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Position (rad)', 'Interpreter', 'latex');
title('Tracking Trajectory of Agent 1 (Initial $q_1=0$)', 'Interpreter', 'latex');
legend('show', 'Location', 'best', 'Interpreter', 'latex');

figure(2); set(gcf, 'Position', [750, 100, 600, 450]);
hold on; box on;
ylim([-0.7, 0.7]); 

phi_t_vals = (rho_0 - rho_inf) * exp(-a_ppc * t_span) + rho_inf;
plot(t_span(plot_idx), phi_t_vals(plot_idx), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Upper Bound');
plot(t_span(plot_idx), -phi_t_vals(plot_idx), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Lower Bound');

plot(t_span(plot_idx), z1_err(plot_idx), 'k-', 'LineWidth', lw, 'DisplayName', 'Tracking Error $z_{1,1}$');
plot([0, T], [0, 0], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');

xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Tracking Error $z_{1,1}$', 'Interpreter', 'latex');
title('Tracking Error under Traditional BLF Control', 'Interpreter', 'latex');
legend('show', 'Location', 'northeast', 'Interpreter', 'latex');


%% --- 4. 导出绘图数据为 MAT 格式  ---
fprintf('正在导出纯轨迹与误差数据...\n');

time_data = t_span(plot_idx);
leader_trajectory = v1_ref(plot_idx);
agent_trajectory = q1_act(plot_idx);
tracking_error = z1_err(plot_idx);

filename = 'TrajectoryAndErrorData_BLF.mat';
save(filename, 'time_data', 'leader_trajectory', 'agent_trajectory', 'tracking_error');


function dY = derivatives_combined(t, Y, u_cont, N, n, l, scrS0, ...
    psi0, rho_0, rho_inf, a_ppc, k_a, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, Lambda_W, gamma_W, ...
    p_i, K_gain, g_grav)

    [v, q, q_dot, W_hat] = extract_all_states(Y, N, n, l);
    v_dot = scrS0 * v;
    q_ddot = zeros(n, N);
    
    for i = 1:N
        phi_i_val = get_phi_attack(i, t);
        [delta_i1, delta_i2] = get_disturbances(i, q(:,i), q_dot(:,i), t);
        tau_act = phi_i_val*u_cont(:,i) + delta_i1 + delta_i2;
        
        p = p_i(i);
        M_mat = [p.p1+2*p.p2*cos(q(2,i)), p.p3+p.p2*cos(q(2,i));
                 p.p3+p.p2*cos(q(2,i)), p.p3];
        C_mat = [-p.p2*q_dot(2,i)*sin(q(2,i)), -p.p2*(q_dot(1,i)+q_dot(2,i))*sin(q(2,i));
                 p.p2*q_dot(1,i)*sin(q(2,i)), 0];
        G_mat = [p.p4*g_grav*cos(q(1,i))+p.p5*g_grav*cos(q(1,i)+q(2,i));
                 p.p5*g_grav*cos(q(1,i)+q(2,i))];
        
        q_ddot(:,i) = M_mat \ (tau_act - C_mat*q_dot(:,i) - G_mat);
        
        % 计算虚拟控制器
        alpha_i = compute_alpha(q(:,i), v, v_dot, psi0(i), t, rho_0, rho_inf, a_ppc, k_a, K_gain(i));
        beta2 = q_dot(:,i) - alpha_i;
        
        % 数值微分
        dt_small = 1e-6;
        v_next = v + v_dot*dt_small;
        v_dot_next = scrS0 * v_next; 
        alpha_next = compute_alpha(q(:,i), v_next, v_dot_next, psi0(i), t+dt_small, rho_0, rho_inf, a_ppc, k_a, K_gain(i));
        alpha_dot_i = (alpha_next - alpha_i)/dt_small;
        
        Z_i = [q(:,i); q_dot(:,i); alpha_i; alpha_dot_i];
        
        rbf_centers = linspace(-1, 1, l);
        S_i_vec = exp(-sum((Z_i - rbf_centers).^2, 1)' / (2*rbf_width^2));
        
        Pi_i = compute_switching(q(:,i), q_dot(:,i), alpha_i, d_switch_1, d_switch_2, r_switch);

        W_hat_dot = Lambda_W * (S_i_vec * (beta2' * Pi_i) - gamma_W * W_hat);
    end
    
    dY = [v_dot; 
          reshape(q_dot,[],1); 
          reshape(q_ddot,[],1);
          reshape(W_hat_dot,[],1)];
end

function [u_cont, alpha_cont] = calculate_control(Y, t, N, n, l, rho_0, rho_inf, a_ppc, k_a, rbf_width, varpi, d_switch_1, d_switch_2, ...
    r_switch, p_i, K_gain, g, Phi_min, psi0, scrS0)

    [v, q, q_dot, W_hat] = extract_all_states(Y, N, n, l);
    v_dot = scrS0 * v;
    u_cont = zeros(n, N); 
    alpha_cont = zeros(n, N);
    
    for i = 1:N
        [u_cont(:,i), alpha_cont(:,i)] = compute_single_control(i, q(:,i), q_dot(:,i), v, v_dot, scrS0, psi0(i), ...
            W_hat, t, rho_0, rho_inf, a_ppc, k_a, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, l, K_gain(i), Phi_min(i));
    end
end

function [u_i, alpha_i] = compute_single_control(i, q_i, q_dot_i, v, v_dot, scrS0, psi0_i, ...
    W_hat, t, rho_0, rho_inf, a_ppc, k_a, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, l, K_gain_i, Phi_min_i)
    alpha_i = compute_alpha(q_i, v, v_dot, psi0_i, t, rho_0, rho_inf, a_ppc, k_a, K_gain_i);
    
    dt_small = 1e-6;
    v_next = v + v_dot*dt_small;
    v_dot_next = scrS0 * v_next;
    alpha_next = compute_alpha(q_i, v_next, v_dot_next, psi0_i, t+dt_small, rho_0, rho_inf, a_ppc, k_a, K_gain_i);
    alpha_dot_i = (alpha_next - alpha_i) / dt_small;
    
    beta2 = q_dot_i - alpha_i;
    Z_i = [q_i; q_dot_i; alpha_i; alpha_dot_i];
    
    rbf_centers = linspace(-1, 1, l);
    S_i = exp(-sum((Z_i - rbf_centers).^2, 1)' / (2*rbf_width^2));
    
    Pi_i = compute_switching(q_i, q_dot_i, alpha_i, d_switch_1, d_switch_2, r_switch);
    
    e = q_i - psi0_i * v;
    phi_t = (rho_0 - rho_inf)*exp(-a_ppc*t) + rho_inf;
    P = e ./ (phi_t^2 - e.^2 + 1e-6); 
    
    % 鲁棒项计算
    norm_Z = norm(Z_i);
    f_known_bound = 20 + 7*norm_Z + 3*norm_Z^2; 
    f_im_U = [f_known_bound; f_known_bound];
    
    Z2_im = zeros(2,1);
    for m = 1:2
        Z2_im(m) = f_im_U(m) * tanh( (f_im_U(m) * beta2(m)) / varpi );
    end

    Phi_a = W_hat' * S_i; 
    Phi_b = Z2_im;
    
    u_i2 = -P - Pi_i * Phi_a - (eye(2) - Pi_i) * Phi_b; 
    u_i = -K_gain_i.K2 * beta2 + u_i2 / Phi_min_i;
end

function alpha = compute_alpha(q_i, v, v_dot, psi0_i, t, rho_0, rho_inf, a_ppc, k_a, K_gain_i)
    ref = psi0_i * v;
    ref_dot = psi0_i * v_dot;
    e = q_i - ref; 
    
    phi_t = (rho_0 - rho_inf)*exp(-a_ppc*t) + rho_inf;
    phi_dot = -a_ppc*(rho_0 - rho_inf)*exp(-a_ppc*t);

    sigma_t = sqrt( 2*(phi_dot/phi_t)^2 + k_a );
    
    alpha = ref_dot - K_gain_i.K1 * e + sigma_t * e;
end

function Pi_i = compute_switching(q_i, q_dot_i, alpha_i, d_switch_1, d_switch_2, r_switch)
    n = length(q_i);
    pi_diag = zeros(n, 1);
    
    for m = 1:n
        states_m = [q_i(m); q_dot_i(m); alpha_i(m)];
        val_prod = 1;
        for k = 1:3 
            z_val = abs(states_m(k));
            if k == 1
                d1 = d_switch_1(1); d2 = d_switch_2(1);
            else 
                d1 = d_switch_1(2); d2 = d_switch_2(2);
            end
            
            if z_val < d1
                m_func = 1;
            elseif z_val > d2
                m_func = 0;
            else
                term = (z_val^2 - d1^2) / (r_switch * (d2^2 - d1^2));
                m_func = ((d2^2 - z_val^2) / (d2^2 - d1^2)) * exp(-term^2);
            end
            val_prod = val_prod * m_func;
        end
        pi_diag(m) = val_prod;
    end
    Pi_i = diag(pi_diag);
end

function [v, q, q_dot, W_hat] = extract_all_states(Y, N, n, l)
    idx = 1;
    v = Y(idx:idx+n-1); idx=idx+n;
    q = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    q_dot = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    W_hat = reshape(Y(idx:end), [l, n]);
end

function phi_i = get_phi_attack(i, t)
    phi_i = 1;
end

function [delta_i1, delta_i2] = get_disturbances(i, q_i, q_dot_i, t)
    delta_i1 = [0; 0];
    delta_i2 = [0; 0];
end