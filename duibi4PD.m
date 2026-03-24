clear all; close all; clc; warning off;

%% --- 1. 系统参数与初始化 ---
dt = 0.0001; T = 20; t_span = 0:dt:T; N_steps = length(t_span);
N = 1; 
n = 2;

psi0 = [1]; 

scrS0 = [0, 1; -1, 0]; 

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

% PD 控制器增益设计 (Kp 比例增益, Kd 微分增益)
Kp = diag([75, 75]);
Kd = diag([25, 25]);

fprintf('单智能体一般PD控制仿真开始...\n');
update_interval = 20000;

v_hist = zeros(n, N_steps);
q_hist = zeros(n, N_steps, N);
q_dot_hist = zeros(n, N_steps, N);
u_hist = zeros(n, N_steps, N);

v_hist(:,1) = [0.2; 0.1]; 

q_hist(:,1,1) = [0.0; 0.0]; 
q_dot_hist(:,1,1) = [0.0; 0.0];

Y = [v_hist(:,1);
     q_hist(:,1,1);
     q_dot_hist(:,1,1)];

%% --- 2. 仿真主循环 ---
for k = 1:N_steps-1
    t_curr = t_span(k);
    
    v = Y(1:2);
    q = Y(3:4);
    q_dot = Y(5:6);
    
    v_dot = scrS0 * v;
    
    q_ref = psi0 * v;
    q_ref_dot = psi0 * v_dot;
    
    e = q - q_ref;
    e_dot = q_dot - q_ref_dot;
    
    % 动力学建模 
    p = p_i(1);
    M_mat = [p.p1+2*p.p2*cos(q(2)), p.p3+p.p2*cos(q(2));
             p.p3+p.p2*cos(q(2)), p.p3];
    C_mat = [-p.p2*q_dot(2)*sin(q(2)), -p.p2*(q_dot(1)+q_dot(2))*sin(q(2));
             p.p2*q_dot(1)*sin(q(2)), 0];
    G_mat = [p.p4*g_grav*cos(q(1))+p.p5*g_grav*cos(q(1)+q(2));
             p.p5*g_grav*cos(q(1)+q(2))];
             
    % --- 一般 PD 控制器  ---
    u = -Kp * e - Kd * e_dot + G_mat;
    
    q_ddot = M_mat \ (u - C_mat*q_dot - G_mat);
    
    dY = [v_dot; q_dot; q_ddot];
    Y = Y + dt * dY;
    v_hist(:,k+1) = Y(1:2);
    q_hist(:,k+1,1) = Y(3:4);
    q_dot_hist(:,k+1,1) = Y(5:6);
    u_hist(:,k+1,1) = u;
    
    if mod(k, update_interval) == 0 
        fprintf('当前进度: %.1f%%\n', k / (N_steps-1) * 100);
    end
end
fprintf('仿真完成，开始绘图...\n');

%% --- 3. 绘图部分 ---
v1_ref = psi0 * squeeze(v_hist(1,:));
q1_act = squeeze(q_hist(1,:,1));
z1_err = q1_act - v1_ref;

plot_idx = 1:500:N_steps;
lw = 1.5; 

figure(1); set(gcf, 'Position', [100, 100, 600, 450]);
hold on; box on;

plot(t_span(plot_idx), q1_act(plot_idx), 'b-', 'LineWidth', lw, 'DisplayName', 'Actual Trajectory $q_{1,1}$');
plot(t_span(plot_idx), v1_ref(plot_idx), 'r--', 'LineWidth', lw, 'DisplayName', 'Leader Trajectory $v_1$');

xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Position (rad)', 'Interpreter', 'latex');
title('Tracking Trajectory of Agent 1 (Initial $q_1=0$)', 'Interpreter', 'latex');
legend('show', 'Location', 'best', 'Interpreter', 'latex');


figure(2); set(gcf, 'Position', [750, 100, 600, 450]);
hold on; box on;

plot(t_span(plot_idx), z1_err(plot_idx), 'k-', 'LineWidth', lw, 'DisplayName', 'Tracking Error $z_{1,1}$');

plot([0, T], [0, 0], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');

xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Tracking Error $z_{1,1}$', 'Interpreter', 'latex');
title('Tracking Error under General PD Control', 'Interpreter', 'latex');
legend('show', 'Location', 'best', 'Interpreter', 'latex');

%% --- 4. 导出绘图数据为 MAT 格式 ---
fprintf('正在导出绘图数据...\n');

time_data = t_span(plot_idx);
leader_trajectory = v1_ref(plot_idx);
agent_trajectory = q1_act(plot_idx);
tracking_error = z1_err(plot_idx);
filename = 'TrajectoryAndErrorData.mat';
save(filename, 'time_data', 'leader_trajectory', 'agent_trajectory', 'tracking_error');
