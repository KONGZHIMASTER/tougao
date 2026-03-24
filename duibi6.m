clear all; close all; clc; warning off;

%% --- 1. 系统参数与初始化 ---
dt = 0.0001; T = 20; t_span = 0:dt:T; N_steps = length(t_span);
N = 1; 
n = 2;
psi0 = [1]; 

% 领导者系统矩阵
scrS0 = [0, 1; -1, 0]; 

b_s = 0.9996; 
b_f = 0.02; 
a_ppc = 1.0;

l_nn = 256;
rbf_width = 1.0;
rng(42); % 固定随机种子，确保每次中心一致
rbf_centers = 4 * rand(8, l_nn) - 2; 
W_hat_0_norm = norm(0.1 * ones(l_nn, n), 'fro'); % 初始理想权值范数经验值
Lambda_Q = 1.00; % Q_hat 的更新率
mu_Q = 0.5;      % 泄漏参数
varpi = 0.2; 

% 平滑切换函数边界
d_switch_1 = [2, 2, 15, 150]; 
d_switch_2 = [4, 4, 30, 300]; 
r_switch = 1.25;

g_grav = 9.81;

% 机械臂物理参数
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

% 控制器增益
K_gain = repmat(struct('K1',diag([2.5,2.5]),'K2',diag([75,75])),N,1);

fprintf('单智能体仿真开始 (全局GPPF方法验证, 初始位置 q1=0)...\n');
update_interval = 20000;

% 状态变量历史记录初始化
v_hist = zeros(n, N_steps);
q_hist = zeros(n, N_steps, N);
q_dot_hist = zeros(n, N_steps, N);
Q_hat_hist = zeros(N, N_steps); 
u_hist = zeros(n, N_steps, N);
alpha_hist = zeros(n, N_steps, N); 

% 初始条件
v_hist(:,1) = [0.2; 0.1]; 

q_hist(:,1,1) = [0.0; 0.0]; 
q_dot_hist(:,1,1) = [0.0; 0.0];
Q_hat_hist(:,1) = 0; 

Y = [v_hist(:,1);
     reshape(q_hist(:,1,:),[],1);
     reshape(q_dot_hist(:,1,:),[],1);
     Q_hat_hist(:,1)];

%% --- 2. 仿真主循环 ---
for k = 1:N_steps-1
    t_curr = t_span(k);
    
    [u_cont, alpha_cont] = calculate_control(Y, t_curr, N, n, b_s, b_f, a_ppc, ...
        W_hat_0_norm, rbf_centers, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, K_gain, psi0, scrS0);
    
    u_hist(:,k,:) = u_cont;
    alpha_hist(:,k,:) = alpha_cont;
    
    dY = derivatives_combined(t_curr, Y, u_cont, N, n, scrS0, psi0, ...
        b_s, b_f, a_ppc, W_hat_0_norm, rbf_centers, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, ...
        Lambda_Q, mu_Q, p_i, K_gain, g_grav);
          
    Y = Y + dt * dY;
    
    [v_new, q_new, q_dot_new, Q_hat_new] = extract_all_states(Y, N, n);
    
    v_hist(:,k+1) = v_new;
    q_hist(:,k+1,:) = q_new;
    q_dot_hist(:,k+1,:) = q_dot_new;
    Q_hat_hist(:,k+1) = Q_hat_new;
    
    if mod(k, update_interval) == 0 
        fprintf('当前进度: %.1f%%\n', k / (N_steps-1) * 100);
    end
end
u_hist(:,N_steps,:) = u_hist(:,N_steps-1,:);
alpha_hist(:,N_steps,:) = alpha_hist(:,N_steps-1,:);

fprintf('仿真完成，开始绘图并导出数据...\n');

%% --- 3. 绘图部分  ---

v1_ref = psi0(1) * squeeze(v_hist(1,:));
q1_act = squeeze(q_hist(1,:,1));
z1_err = q1_act - v1_ref;

plot_idx = 1:500:N_steps;
lw = 1.5; 

% ==================== 窗口 1: 跟踪轨迹 ====================
figure(1); set(gcf, 'Position', [100, 100, 600, 450]);
hold on; box on;

plot(t_span(plot_idx), q1_act(plot_idx), 'b-', 'LineWidth', lw, 'DisplayName', 'Actual Trajectory $q_{1,1}$');
plot(t_span(plot_idx), v1_ref(plot_idx), 'r--', 'LineWidth', lw, 'DisplayName', 'Leader Trajectory $v_{1}$');

xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Position (rad)', 'Interpreter', 'latex');
title('Tracking Trajectory of Agent 1 (Initial $q_1=0$)', 'Interpreter', 'latex');
legend('show', 'Location', 'best', 'Interpreter', 'latex');

% ==================== 窗口 2: 误差轨迹与GPPF边界 ====================
figure(2); set(gcf, 'Position', [750, 100, 600, 450]);
hold on; box on;
ylim([-1.0, 1.0]); 

U_bound = zeros(1, N_steps);
for k = 1:N_steps
    gamma_t = (1-b_f)*exp(-a_ppc*t_span(k)) + b_f;
    if k == 1
        U_bound(k) = NaN; % 
    else
        U_bound(k) = gamma_t * sqrt(b_s) / sqrt(1 - gamma_t^2);
    end
end

% 绘制 GPPF 边界
plot(t_span(plot_idx), U_bound(plot_idx), 'm--', 'LineWidth', 1.5, 'DisplayName', 'GPPF Upper Bound');
plot(t_span(plot_idx), -U_bound(plot_idx), 'b--', 'LineWidth', 1.5, 'DisplayName', 'GPPF Lower Bound');

% 绘制实际误差
plot(t_span(plot_idx), z1_err(plot_idx), 'k-', 'LineWidth', lw, 'DisplayName', 'Tracking Error $z_{1,1}$');
plot([0, T], [0, 0], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');

xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Tracking Error $z_{1,1}$', 'Interpreter', 'latex');
title('Tracking Error under Proposed GPPF Control', 'Interpreter', 'latex');
legend('show', 'Location', 'northeast', 'Interpreter', 'latex');

%% --- 4. 导出绘图数据为 MAT 格式 (不含边界) ---
fprintf('正在导出纯轨迹与误差数据...\n');

time_data = t_span(plot_idx);
leader_trajectory = v1_ref(plot_idx);
agent_trajectory = q1_act(plot_idx);
tracking_error = z1_err(plot_idx);

filename = 'TrajectoryAndErrorData_GPPF.mat';

save(filename, 'time_data', 'leader_trajectory', 'agent_trajectory', 'tracking_error');

%% --- 5. 核心控制函数 (您的手稿版本) ---

function dY = derivatives_combined(t, Y, u_cont, N, n, scrS0, psi0, ...
    b_s, b_f, a_ppc, W_hat_0_norm, rbf_centers, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, ...
    Lambda_Q, mu_Q, p_i, K_gain, g_grav)

    [v, q, q_dot, Q_hat] = extract_all_states(Y, N, n);
    v_dot = scrS0 * v;
    v_ddot = scrS0 * v_dot;
    q_ddot = zeros(n, N);
    Q_hat_dot = zeros(N, 1);
    
    for i = 1:N
        p = p_i(i);
        M_mat = [p.p1+2*p.p2*cos(q(2,i)), p.p3+p.p2*cos(q(2,i));
                 p.p3+p.p2*cos(q(2,i)), p.p3];
        C_mat = [-p.p2*q_dot(2,i)*sin(q(2,i)), -p.p2*(q_dot(1,i)+q_dot(2,i))*sin(q(2,i));
                 p.p2*q_dot(1,i)*sin(q(2,i)), 0];
        G_mat = [p.p4*g_grav*cos(q(1,i))+p.p5*g_grav*cos(q(1,i)+q(2,i));
                 p.p5*g_grav*cos(q(1,i)+q(2,i))];
        
        q_ddot(:,i) = M_mat \ (u_cont(:,i) - C_mat*q_dot(:,i) - G_mat);
        
        q_ref = psi0(i) * v;
        q_ref_dot = psi0(i) * v_dot;
        q_ref_ddot = psi0(i) * v_ddot;
        
        [alpha_i, alpha_dot_i] = compute_alpha_exact(q(:,i), q_dot(:,i), q_ref, q_ref_dot, q_ref_ddot, t, b_s, b_f, a_ppc, K_gain(i));
        beta2 = q_dot(:,i) - alpha_i;

        Z_i = [q(:,i); q_dot(:,i); alpha_i; alpha_dot_i];
        S_i_vec = exp(-sum((Z_i - rbf_centers).^2, 1)' / (2*rbf_width^2));
        S_i_norm = norm(S_i_vec);
        
        Pi_i = compute_switching(Z_i, d_switch_1, d_switch_2, r_switch);

        sgn_beta2 = sat(beta2 / varpi); 
        term_scalar = (beta2' * Pi_i * sgn_beta2); 
        regressor = term_scalar * S_i_norm;
        Q_hat_dot(i) = Lambda_Q * (regressor - mu_Q * Q_hat(i));
    end
    
    dY = [v_dot; 
          reshape(q_dot,[],1); 
          reshape(q_ddot,[],1);
          Q_hat_dot];
end

function [u_cont, alpha_cont] = calculate_control(Y, t, N, n, b_s, b_f, a_ppc, ...
    W_hat_0_norm, rbf_centers, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, K_gain, psi0, scrS0)

    [v, q, q_dot, Q_hat] = extract_all_states(Y, N, n);
    v_dot = scrS0 * v;
    v_ddot = scrS0 * v_dot;
    u_cont = zeros(n, N); 
    alpha_cont = zeros(n, N);
    
    for i = 1:N
        q_ref = psi0(i) * v;
        q_ref_dot = psi0(i) * v_dot;
        q_ref_ddot = psi0(i) * v_ddot;
        
        [u_cont(:,i), alpha_cont(:,i)] = compute_single_control(q(:,i), q_dot(:,i), q_ref, q_ref_dot, q_ref_ddot, ...
            Q_hat(i), W_hat_0_norm, rbf_centers, t, b_s, b_f, a_ppc, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, K_gain(i));
    end
end

function [u_i, alpha_i] = compute_single_control(q_i, q_dot_i, q_ref, q_ref_dot, q_ref_ddot, ...
    Q_hat_i, W_hat_0_norm, rbf_centers, t, b_s, b_f, a_ppc, rbf_width, varpi, d_switch_1, d_switch_2, r_switch, K_gain_i)

    [alpha_i, alpha_dot_i] = compute_alpha_exact(q_i, q_dot_i, q_ref, q_ref_dot, q_ref_ddot, t, b_s, b_f, a_ppc, K_gain_i);
    beta2 = q_dot_i - alpha_i;
    Z_i = [q_i; q_dot_i; alpha_i; alpha_dot_i];

    S_i_vec = exp(-sum((Z_i - rbf_centers).^2, 1)' / (2*rbf_width^2));
    S_i_norm = norm(S_i_vec);
    
    Pi_i = compute_switching(Z_i, d_switch_1, d_switch_2, r_switch);

    norm_Z = norm(Z_i);
    F_upper = [20 + 7*norm_Z + 3*norm_Z^2; 20 + 7*norm_Z + 3*norm_Z^2]; 
    omega_robust = zeros(2,1);
    for m = 1:2
        omega_robust(m) = F_upper(m) * tanh( (F_upper(m) * beta2(m)) / varpi );
    end

    sgn_beta2 = sat(beta2 / varpi);
    omega_NN_raw = (Q_hat_i + W_hat_0_norm) * S_i_norm * sgn_beta2;
    omega_NN = Pi_i * omega_NN_raw;
    
    % 实际控制输入 (假设 Phi_min = 1)
    u_i = -K_gain_i.K2 * beta2 - omega_NN - (eye(2) - Pi_i) * omega_robust;
end

function [alpha, alpha_dot] = compute_alpha_exact(q_i, q_dot_i, q_ref, q_ref_dot, q_ref_ddot, t, b_s, b_f, a, K_gain_i)
    z_i = q_i - q_ref;
    z_dot_i = q_dot_i - q_ref_dot;
    
    gamma_t = (1-b_f)*exp(-a*t) + b_f;
    gamma_dot = -a*(1-b_f)*exp(-a*t);
    gamma_ddot = a^2*(1-b_f)*exp(-a*t);

    g = 1 / gamma_t;
    g_dot = -gamma_dot / (gamma_t^2);
    g_ddot = -(gamma_ddot*gamma_t - 2*gamma_dot^2) / (gamma_t^3);

    S = z_i.^2 + b_s;
    S_dot = 2 .* z_i .* z_dot_i;
    S_sq = sqrt(S);

    gamma_z = z_i ./ S_sq;
    sigma_i = b_s ./ (S.^(1.5));
    gamma_z_dot = sigma_i .* z_dot_i; 

    rho_i = g .* gamma_z;
    rho_dot_i = g_dot .* gamma_z + g .* gamma_z_dot;

    aleph_i = rho_i ./ (1 - rho_i.^2);
    varphi_i = (1 + rho_i.^2) ./ ((1 - rho_i.^2).^2);
    aleph_dot_i = rho_dot_i .* varphi_i;

    varphi_dot = (2 .* rho_i .* rho_dot_i .* (3 + rho_i.^2)) ./ ((1 - rho_i.^2).^3);
    sigma_dot = -(1.5 * b_s .* S_dot) ./ (S.^(2.5));

    X_i1 = varphi_i .* sigma_i .* g;
    X_i1_dot = varphi_dot .* sigma_i .* g + varphi_i .* sigma_dot .* g + varphi_i .* sigma_i .* g_dot;

    X_i2 = varphi_i .* g_dot .* gamma_z;
    X_i2_dot = varphi_dot .* g_dot .* gamma_z + varphi_i .* g_ddot .* gamma_z + varphi_i .* g_dot .* gamma_z_dot;

    alpha = -X_i2 ./ X_i1 - K_gain_i.K1 * (X_i1 .* aleph_i) + q_ref_dot;

    term1_dot = -(X_i2_dot .* X_i1 - X_i2 .* X_i1_dot) ./ (X_i1.^2);
    term2_dot = - K_gain_i.K1 * (X_i1_dot .* aleph_i + X_i1 .* aleph_dot_i);
    alpha_dot = term1_dot + term2_dot + q_ref_ddot;
end

function Pi_i = compute_switching(Z_i, d_switch_1, d_switch_2, r_switch)
    n = 2;
    d1_full = [d_switch_1(1)*ones(n,1); d_switch_1(2)*ones(n,1); d_switch_1(3)*ones(n,1); d_switch_1(4)*ones(n,1)];
    d2_full = [d_switch_2(1)*ones(n,1); d_switch_2(2)*ones(n,1); d_switch_2(3)*ones(n,1); d_switch_2(4)*ones(n,1)];

    val_prod = 1;
    for k = 1:8
        z_val = abs(Z_i(k));
        d1 = d1_full(k); 
        d2 = d2_full(k);
        
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
    Pi_i = diag(val_prod * ones(n, 1));
end

function [v, q, q_dot, Q_hat] = extract_all_states(Y, N, n)
    idx = 1;
    v = Y(idx:idx+n-1); idx=idx+n;
    q = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    q_dot = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    Q_hat = Y(idx:end);
end

function y = sat(x)
    y = x; 
    y(x > 1) = 1; 
    y(x < -1) = -1;
end