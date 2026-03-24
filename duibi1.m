clear all; close all; clc; warning off;

%% --- 1. 系统参数与初始化 ---
dt = 0.0001; T = 20; t_span = 0:dt:T; N_steps = length(t_span);
N = 4; n = 2;

% 拓扑参数
w_i = [1, -1, -1, 1]; 
A_topo = [0, 0, 0, 0; -1.0, 0, 0, 0; 0, 0, 0, 0; 0, 0, -1.0, 0];
B_topo = diag([1, 0, -1, 0]); 

% 观测器参数 
mu_xi = 5.0; 
E_0 = [0, 1; -1, 0]; R_0 = eye(n);
mu_E = 5.0; mu_R = 5.0;

% --- 欺骗攻击模型  ---
Deception_Attacks = cell(N, 1);
Deception_Attacks{1} = @(t) (t > 10) .* [0.1*sin(1.5*t); 0.1*sin(1.5*t)];
Deception_Attacks{2} = @(t) (t > 12) .* [0; -0.2];
Deception_Attacks{3} = @(t) (t > 15) .* [0.1*sin(0.1*t); 0.1*sin(0.1*t)];
Deception_Attacks{4} = @(t) (t > 10) .* [0; 0];

% --- 状态初始化 ---
v0_hist = zeros(n, N_steps); v0_hist(:,1) = [0.1; 0.2]; 
eta_hist = zeros(n, N_steps, N);
E_hat_hist = zeros(n, n, N_steps, N);
R_hat_hist = zeros(n, n, N_steps, N);

base_mat = 0.2 * ones(n,n);

Init_Errors_Obs = [ 0.0,  0.0,  0.0,  0.0; 
                   -0.0, -0.0, -0.0, -0.0];

for i = 1:N
    eta_hist(:,1,i) =  Init_Errors_Obs(:,i);
    E_hat_hist(:,:,1,i) = i * base_mat;
    R_hat_hist(:,:,1,i) = i * base_mat;
end

% 整合状态向量 Y (去除了 xi 和 d_hat)
Y = [v0_hist(:,1);
     reshape(eta_hist(:,1,:),[],1);
     reshape(E_hat_hist(:,:,1,:), [], 1);
     reshape(R_hat_hist(:,:,1,:), [], 1)];

fprintf('无滤波对比实验仿真开始...\n');

%% --- 2. 仿真主循环 ---
for k = 1:N_steps-1
    t_curr = t_span(k);
    
    % 调用新版的核心函数
    dY = derivatives_observer_nofilter(t_curr, Y, N, n, E_0, R_0, ...
          mu_xi, A_topo, B_topo, Deception_Attacks, mu_E, mu_R);
    
    Y = Y + dt * dY;
    
    [v0_new, eta_new, E_hat_new, R_hat_new] = extract_states_nofilter(Y, N, n);
    
    v0_hist(:,k+1) = v0_new;
    eta_hist(:,k+1,:) = eta_new;
    E_hat_hist(:,:,k+1,:) = E_hat_new;
    R_hat_hist(:,:,k+1,:) = R_hat_new;
    
    if mod(k, 5000) == 0, fprintf('进度: %.1f%%\n', k/(N_steps-1)*100); end
end
fprintf('仿真完成。\n');

%% --- 3. 绘图部分  ---
fig_pos = [100, 100, 560, 420];
c_lines = lines(N);           
lw = 0.5;                      

figure(1); set(gcf, 'Position', fig_pos);

subplot(2,2,1); hold on; box on; set(gca, 'FontName', 'Times New Roman');
for i = 1:N
    plot(t_span, squeeze(eta_hist(1,:,i)), '-', 'Color', c_lines(i,:), 'LineWidth', lw, 'DisplayName', sprintf('A%d (\\eta_{%d,1})', i, i));
end
xlabel('Time (s)'); ylabel('State \eta_1');
title('States \eta in Dim 1 (No Filter)');
legend('show', 'Location', 'best', 'FontName', 'Times New Roman');

subplot(2,2,3); hold on; box on; set(gca, 'FontName', 'Times New Roman');
for i = 1:N
    err_eta_1 = squeeze(eta_hist(1,:,i)) - w_i(i) * v0_hist(1,:);
    plot(t_span, err_eta_1, '-', 'Color', c_lines(i,:), 'LineWidth', lw, 'DisplayName', sprintf('A%d (\\tilde{\\eta}_{%d,1})', i, i));
end
xlabel('Time (s)'); ylabel('Error \tilde{\eta}_1');
title('Errors \tilde{\eta} in Dim 1 (No Filter)');
legend('show', 'Location', 'best', 'FontName', 'Times New Roman');

subplot(2,2,2); hold on; box on; set(gca, 'FontName', 'Times New Roman');
for i = 1:N
    plot(t_span, squeeze(eta_hist(2,:,i)), '-', 'Color', c_lines(i,:), 'LineWidth', lw, 'DisplayName', sprintf('A%d (\\eta_{%d,2})', i, i));
end
xlabel('Time (s)'); ylabel('State \eta_2');
title('States \eta in Dim 2 (No Filter)');
legend('show', 'Location', 'best', 'FontName', 'Times New Roman');

subplot(2,2,4); hold on; box on; set(gca, 'FontName', 'Times New Roman');
for i = 1:N
    err_eta_2 = squeeze(eta_hist(2,:,i)) - w_i(i) * v0_hist(2,:);
    plot(t_span, err_eta_2, '-', 'Color', c_lines(i,:), 'LineWidth', lw, 'DisplayName', sprintf('A%d (\\tilde{\\eta}_{%d,2})', i, i));
end
xlabel('Time (s)'); ylabel('Error \tilde{\eta}_2');
title('Errors \tilde{\eta} in Dim 2 (No Filter)');
legend('show', 'Location', 'best', 'FontName', 'Times New Roman');


%% --- 4. 核心函数 ---
function dY = derivatives_observer_nofilter(t, Y, N, n, E_0, R_0, ...
    mu_xi, A_topo, B_topo, Attacks, mu_E, mu_R)

    [v_0, eta, E_hat, R_hat] = extract_states_nofilter(Y, N, n);
    
    v0_dot = E_0 * v_0;
    
    E_hat_dot = zeros(n, n, N); R_hat_dot = zeros(n, n, N);
    eta_dot = zeros(n, N);

    for i = 1:N
        sum_E = zeros(n,n); sum_R = zeros(n,n);
        coupling_sum = zeros(n,1);
        
        curr_attack = Attacks{i}(t); 
        
        % --- 处理领导者连接 ---
        if B_topo(i,i) ~= 0
            w = B_topo(i,i);
            psi_edge = sign(w);
            
            % 矩阵观测器
            sum_E = sum_E + abs(w) * (E_0 - E_hat(:,:,i));
            sum_R = sum_R + abs(w) * (R_0 - R_hat(:,:,i));
            
            % 包含攻击的直接通信误差
            ref = v_0 + curr_attack;
            term = mu_xi * psi_edge * w * (psi_edge * ref - eta(:,i));
            coupling_sum = coupling_sum + term;
        end
        
        % --- 处理邻居连接 ---
        for j = 1:N
            if A_topo(i,j) ~= 0
                w = A_topo(i,j);
                psi_edge = sign(w);
                
                % 矩阵观测器
                sum_E = sum_E + abs(w) * (E_hat(:,:,j) - E_hat(:,:,i));
                sum_R = sum_R + abs(w) * (R_hat(:,:,j) - R_hat(:,:,i));
                
                % 包含攻击的直接通信误差
                ref = eta(:,j) + curr_attack;
                term = mu_xi * psi_edge * w * (psi_edge * ref - eta(:,i));
                coupling_sum = coupling_sum + term;
            end
        end
        
        E_hat_dot(:,:,i) = mu_E * sum_E;
        R_hat_dot(:,:,i) = mu_R * sum_R;
        
        % 直接由 E_hat 驱动并加上含有攻击的误差总和
        eta_dot(:,i) = E_hat(:,:,i) * eta(:,i) + coupling_sum; 
    end
    
    dY = [v0_dot; reshape(eta_dot,[],1); reshape(E_hat_dot,[],1); reshape(R_hat_dot,[],1)];
end

function [v_0, eta, E_hat, R_hat] = extract_states_nofilter(Y, N, n)
    idx = 1;
    v_0 = Y(idx:idx+n-1); idx=idx+n;
    eta = reshape(Y(idx:idx+n*N-1), [n,N]); idx=idx+n*N;
    E_hat = reshape(Y(idx:idx+n*n*N-1), [n, n, N]); idx=idx+n*n*N;
    R_hat = reshape(Y(idx:idx+n*n*N-1), [n, n, N]); 
end