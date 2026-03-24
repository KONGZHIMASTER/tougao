close all; clear all; clc; warning off;

% time parameters
dt = 0.0001; 
T = 20; 
t_span = 0:dt:T; 
N_steps = length(t_span);
nn = 1;

% system parameters
N = 4; 
n = 2;
g_grav = 9.81;

% mechanical parameters matrix P_params[p1, p2, p3, p4, p5]
P_params = zeros(N, 5);
params_raw =[
    1.3, 1.8, 2.0, 2.0, 1.00, 1.00, 0.4333, 0.6000; 
    1.6, 1.3, 2.1, 1.9, 1.05, 0.95, 0.5880, 0.3911; 
    2.0, 1.5, 1.8, 2.2, 0.90, 1.10, 0.5400, 0.6050; 
    1.9, 1.6, 2.1, 2.0, 1.05, 1.00, 0.6983, 0.5333  
];
for i = 1:N
    m1=params_raw(i,1); m2=params_raw(i,2); l1=params_raw(i,3); l2=params_raw(i,4);
    r1=params_raw(i,5); r2=params_raw(i,6); J1=params_raw(i,7); J2=params_raw(i,8);
    P_params(i,1) = m1*r1^2 + m2*(l1^2 + r2^2) + J1 + J2; 
    P_params(i,2) = m2*l1*r2;
    P_params(i,3) = m2*r2^2 + J2; 
    P_params(i,4) = m1*r1 + m2*l1; 
    P_params(i,5) = m2*r2;
end

% network parameters
w_i =[1, -1, -1, 1]; 
A_topo =[0, 0, 0, 0; -1.0, 0, 0, 0; 0, 0, 0, 0; 0, 0, -1.0, 0];
B_topo = diag([1, 0, -1, 0]); 
mu_xi = 5.0; mu_d = 10.0; g_obs = 7.5; s_obs = 1.00;
E_0 = -[0, -1; 1, 0]; 
R_0 = eye(n); 
A_ARE = repmat(0.8198 * eye(n), 1, 1, N);
mu_E = 5.0; mu_R = 5.0; 

% controller parameters
b_s = 0.99; b_f = 0.1; a_ppc = 1.0;
Lambda = 2.50; mu_Q = 0.5; varpi = 0.2; 
d_switch_1 = [2, 2, 15, 150]; 
d_switch_2 = [4, 4, 30, 300]; 
r_switch = 1.25;
phi_min =[3, 1, 1, 1]; 
K1_gain = zeros(2, 2, N);
K2_gain = zeros(2, 2, N);
for i = 1:N
    K1_gain(:,:,i) = diag([2.5, 2.5]);
    if i == 1
        K2_gain(:,:,i) = diag([250, 250]);
    else
        K2_gain(:,:,i) = diag([75, 75]);
    end
end

% ELM / RBF parameters
rng(42); 
l_nn = 256;
rbf_width = 1.0;
rbf_centers = zeros(8, l_nn, N);
W_hat_0 = 0.1 * ones(l_nn, n, N);
for i = 1:N
    rbf_centers(:,:,i) = 4 * rand(8, l_nn) - 2; 
end

% initialization
v0 = [0.1; 0.2];
eta = zeros(n, N); 
xi = zeros(n, N);
d_hat = zeros(N, N+1); 
E_hat = zeros(n, n, N);
R_hat = zeros(n, n, N);
q = zeros(n, N); 
dq = zeros(n, N);
Q_hat = zeros(N, 1);
u_cont = zeros(n, N);

offset_vals =[0.3, -0.3, 0.6, -0.6]; 
Init_Errors_Ctrl =[0.45, 0.15, 0.6, 0.3; 0, 0, 0, 0]; 
for i = 1:N
    eta(:,i) = [0; 0];
    
    E_hat(:,:,i) = [0.5, -0.5; 0.5, 0.5] + offset_vals(i) * ones(n,n);
    R_hat(:,:,i) = [0.8, 0.2; 0.2, 0.8] + offset_vals(i) * ones(n,n);
    
    q(:,i) = R_hat(:,:,i) * eta(:,i) + Init_Errors_Ctrl(:,i); 
end

% data record arrays
X_v0 = zeros(n, N_steps);
X_q = zeros(n, N_steps, N);
X_dq = zeros(n, N_steps, N);
X_u = zeros(n, N_steps, N);
X_Q_hat = zeros(N, N_steps);
X_d_hat = zeros(N, N+1, N_steps);
X_E_hat = zeros(n, n, N_steps, N);
X_xi = zeros(n, N_steps, N);
X_eta = zeros(n, N_steps, N);
X_qd = zeros(n, N_steps, N);
X_z_err = zeros(n, N_steps, N);

% main loop
for nn = 1:N_steps
    t = t_span(nn);
    h_func = g_obs * exp(-s_obs * t);
    
    % derivatives initialization
    dv0 = E_0 * v0;
    dE_hat = zeros(n, n, N);
    dR_hat = zeros(n, n, N);
    ddR_hat = zeros(n, n, N);
    dxi = zeros(n, N); 
    dd_hat = zeros(N, N+1); 
    deta = zeros(n, N);
    ddq = zeros(n, N);
    dQ_hat = zeros(N, 1);
    
    qd = zeros(n, N);
    dqd = zeros(n, N);
    ddqd = zeros(n, N);
    
    % Pass 1: Observer Dynamics & Deception Attacks
    for i = 1:N
        % Deception Attacks
        if i == 1 && t > 10, attack_v =[0.1*sin(1.5*t); 0.1*sin(1.5*t)];
        elseif i == 2 && t > 12, attack_v = [0; -0.2];
        elseif i == 3 && t > 15, attack_v =[0.1*sin(0.1*t); 0.1*sin(0.1*t)];
        else, attack_v =[0; 0]; end
        
        sum_E = zeros(n,n); sum_R = zeros(n,n); xi_accum = zeros(n,1);
        norm_xi_P = norm(xi(:,i)' * A_ARE(:,:,i));
        
        if B_topo(i,i) ~= 0
            w = B_topo(i,i); psi_edge = sign(w);
            sum_E = sum_E + abs(w) * (E_0 - E_hat(:,:,i));
            sum_R = sum_R + abs(w) * (R_0 - R_hat(:,:,i));
            dd_hat(i, N+1) = mu_d * (norm_xi_P - h_func * d_hat(i, N+1));
            d_hat_term = -(A_ARE(:,:,i)' * xi(:,i) * d_hat(i, N+1)) / (norm_xi_P + h_func);
            ref = v0 + attack_v;
            xi_accum = xi_accum + mu_xi * psi_edge * w * (psi_edge * ref - eta(:,i)) + psi_edge * mu_xi * w * d_hat_term;
        end
        for j = 1:N
            if A_topo(i,j) ~= 0
                w = A_topo(i,j); psi_edge = sign(w);
                sum_E = sum_E + abs(w) * (E_hat(:,:,j) - E_hat(:,:,i));
                sum_R = sum_R + abs(w) * (R_hat(:,:,j) - R_hat(:,:,i));
                dd_hat(i, j) = mu_d * (norm_xi_P - h_func * d_hat(i, j));
                d_hat_term = -(A_ARE(:,:,i)' * xi(:,i) * d_hat(i, j)) / (norm_xi_P + h_func);
                ref = eta(:,j) + attack_v;
                xi_accum = xi_accum + mu_xi * psi_edge * w * (psi_edge * ref - eta(:,i)) + psi_edge * mu_xi * w * d_hat_term;
            end
        end
        dE_hat(:,:,i) = mu_E * sum_E;
        dR_hat(:,:,i) = mu_R * sum_R;
        dxi(:,i) = -5*xi(:,i) + xi_accum;
        deta(:,i) = E_hat(:,:,i) * eta(:,i) + xi(:,i);
    end
    
    % Pass 2: Reference generation, Controller, Robot Dynamics
    for i = 1:N
        % calculate R_hat second derivative
        sum_dR = zeros(n,n);
        if B_topo(i,i) ~= 0, sum_dR = sum_dR + abs(B_topo(i,i)) * (0 - dR_hat(:,:,i)); end
        for j = 1:N
            if A_topo(i,j) ~= 0, sum_dR = sum_dR + abs(A_topo(i,j)) * (dR_hat(:,:,j) - dR_hat(:,:,i)); end
        end
        ddR_hat(:,:,i) = mu_R * sum_dR;
        
        % reference states
        qd(:,i) = R_hat(:,:,i) * eta(:,i);
        dqd(:,i) = dR_hat(:,:,i) * eta(:,i) + R_hat(:,:,i) * deta(:,i);
        ddeta_i = dE_hat(:,:,i) * eta(:,i) + E_hat(:,:,i) * deta(:,i) + dxi(:,i);
        ddqd(:,i) = ddR_hat(:,:,i) * eta(:,i) + 2 * dR_hat(:,:,i) * deta(:,i) + R_hat(:,:,i) * ddeta_i;
        
        % Alpha & Prescribed Performance Control variables
        z = q(:,i) - qd(:,i);
        dz = dq(:,i) - dqd(:,i);
        
        gamma_t = (1-b_f)*exp(-a_ppc*t) + b_f;
        dgamma = -a_ppc*(1-b_f)*exp(-a_ppc*t);
        ddgamma = a_ppc^2*(1-b_f)*exp(-a_ppc*t);
        
        g = 1 / gamma_t;
        dg = -dgamma / (gamma_t^2);
        ddg = -(ddgamma*gamma_t - 2*dgamma^2) / (gamma_t^3);
        
        S = z.^2 + b_s;
        dS = 2 .* z .* dz;
        S_sq = sqrt(S);
        gamma_z = z ./ S_sq;
        sigma_z = b_s ./ (S.^(1.5));
        dgamma_z = sigma_z .* dz;
        
        rho = g .* gamma_z;
        drho = dg .* gamma_z + g .* dgamma_z;
        aleph = rho ./ (1 - rho.^2);
        varphi = (1 + rho.^2) ./ ((1 - rho.^2).^2);
        daleph = drho .* varphi;
        dvarphi = (2 .* rho .* drho .* (3 + rho.^2)) ./ ((1 - rho.^2).^3);
        dsigma_z = -(1.5 * b_s .* dS) ./ (S.^(2.5));
        
        X1 = varphi .* sigma_z .* g;
        dX1 = dvarphi .* sigma_z .* g + varphi .* dsigma_z .* g + varphi .* sigma_z .* dg;
        X2 = varphi .* dg .* gamma_z;
        dX2 = dvarphi .* dg .* gamma_z + varphi .* ddg .* gamma_z + varphi .* dg .* dgamma_z;
        
        alpha = -X2 ./ X1 - K1_gain(:,:,i) * diag(X1) * aleph + dqd(:,i);
        dterm1 = -(dX2 .* X1 - X2 .* dX1) ./ (X1.^2);
        dterm2 = - K1_gain(:,:,i) * (diag(dX1) * aleph + diag(X1) * daleph);
        dalpha = dterm1 + dterm2 + ddqd(:,i);
        
        % Switching filter Pi_i
        Z_NN =[q(:,i); dq(:,i); alpha; dalpha];
        d1_full =[d_switch_1(1)*ones(2,1); d_switch_1(2)*ones(2,1); d_switch_1(3)*ones(2,1); d_switch_1(4)*ones(2,1)];
        d2_full =[d_switch_2(1)*ones(2,1); d_switch_2(2)*ones(2,1); d_switch_2(3)*ones(2,1); d_switch_2(4)*ones(2,1)];
        val_prod = 1;
        for k_idx = 1:8
            z_val = abs(Z_NN(k_idx));
            d1_val = d1_full(k_idx); d2_val = d2_full(k_idx);
            if z_val < d1_val
                m_func = 1;
            elseif z_val > d2_val
                m_func = 0;
            else
                m_func = ((d2_val^2 - z_val^2)/(d2_val^2 - d1_val^2)) * exp(-((z_val^2 - d1_val^2)/(r_switch*(d2_val^2 - d1_val^2)))^2);
            end
            val_prod = val_prod * m_func;
        end
        Pi_i = diag(val_prod * ones(2, 1));
        
        % RBF Neural Network & Control Torque tau
        beta2 = dq(:,i) - alpha;
        S_i = exp(-sum((Z_NN - rbf_centers(:,:,i)).^2, 1)' / (2*rbf_width^2));
        norm_Z = norm(Z_NN);
        F_upper =[20 + 7*norm_Z + 3*norm_Z^2; 20 + 7*norm_Z + 3*norm_Z^2];
        omega_robust = zeros(2,1);
        for m_idx = 1:2
            omega_robust(m_idx) = F_upper(m_idx) * tanh((F_upper(m_idx) * beta2(m_idx)) / varpi);
        end
        
        W_hat_norm = norm(W_hat_0(:,:,i), 'fro');
        S_i_norm = norm(S_i);
        sgn_beta2 = beta2 / varpi;
        sgn_beta2(sgn_beta2 > 1) = 1; sgn_beta2(sgn_beta2 < -1) = -1; % sat function inline
        
        omega_NN = Pi_i * ((Q_hat(i) + W_hat_norm) * S_i_norm * sgn_beta2);
        u2 = -(omega_NN + (eye(2) - Pi_i) * omega_robust);
        tau = -K2_gain(:,:,i) / phi_min(i) * beta2 + u2 / phi_min(i);
        u_cont(:,i) = tau;
        
        % Actuator Attacks & Disturbances
        if i == 1 && t > 5
            attack_val = min(3 + 1.5 * exp(0.12 * t), 13);
            Phi_i = diag([attack_val, attack_val]);
            sigma_dis =[3.3 + 2.7 * exp(-0.5 * t); 3.3 + 2.7 * exp(-0.5 * t)];
        else
            Phi_i = eye(2);
            sigma_dis = [0; 0];
        end
        f_dis =[0.5*dq(1,i) + 2*sin(4*q(1,i)) + 1.6*sin(dq(1,i));
                 1.2*dq(1,i) - 1.6*sin(2*q(2,i)) + 0.8*sin(dq(2,i))];
        tau_act = Phi_i * tau + sigma_dis;
        
        % Robot Dynamics calculation M*ddq + C*dq + G = tau_act - f_dis
        p1 = P_params(i,1); p2 = P_params(i,2); p3 = P_params(i,3); p4 = P_params(i,4); p5 = P_params(i,5);
        M =[p1+2*p2*cos(q(2,i)), p3+p2*cos(q(2,i));
             p3+p2*cos(q(2,i)), p3];
        C =[-p2*dq(2,i)*sin(q(2,i)), -p2*(dq(1,i)+dq(2,i))*sin(q(2,i));
             p2*dq(1,i)*sin(q(2,i)), 0];
        G =[p4*g_grav*cos(q(1,i))+p5*g_grav*cos(q(1,i)+q(2,i));
             p5*g_grav*cos(q(1,i)+q(2,i))];
             
        ddq(:,i) = M \ (tau_act - C*dq(:,i) - G - f_dis);
        
        % NN weight parameter update
        regressor = (beta2' * Pi_i * sgn_beta2) * S_i_norm;
        dQ_hat(i) = Lambda * (regressor - mu_Q * Q_hat(i));
    end
    
    % data record
    X_v0(:,nn) = v0;
    X_q(:,nn,:) = q;
    X_dq(:,nn,:) = dq;
    X_u(:,nn,:) = u_cont;
    X_Q_hat(:,nn) = Q_hat;
    X_d_hat(:,:,nn) = d_hat;
    X_E_hat(:,:,nn,:) = E_hat;
    X_xi(:,nn,:) = xi;
    X_eta(:,nn,:) = eta;
    X_qd(:,nn,:) = qd;
    for i = 1:N
        X_z_err(:,nn,i) = q(:,i) - qd(:,i);
    end

    % update parameters (Forward Euler)
    v0 = v0 + dv0 * dt;
    eta = eta + deta * dt;
    xi = xi + dxi * dt;
    d_hat = d_hat + dd_hat * dt;
    E_hat = E_hat + dE_hat * dt;
    R_hat = R_hat + dR_hat * dt;
    q = q + dq * dt;
    dq = dq + ddq * dt;
    Q_hat = Q_hat + dQ_hat * dt;
end

% calculate norms for plotting
U_bound = zeros(1, N_steps);
norm_xi_i = zeros(N, N_steps);
norm_eta_tilde_i = zeros(N, N_steps);
norm_E_tilde_i = zeros(N, N_steps);

for k = 1:N_steps
    gamma_t = (1-b_f)*exp(-a_ppc*t_span(k)) + b_f;
    U_bound(k) = gamma_t * sqrt(b_s) / sqrt(1 - gamma_t^2);
    for i = 1:N
        eta_tilde = X_eta(:,k,i) - w_i(i) * X_v0(:,k);
        norm_E_tilde_i(i,k) = norm(X_E_hat(:,:,k,i) - E_0, 'fro');
        norm_xi_i(i,k) = norm(X_xi(:,k,i));
        norm_eta_tilde_i(i,k) = norm(eta_tilde);
    end
end

% Plots
plot_step = 200; 
idx = 1:plot_step:N_steps; 

% ========================================================
% --- Figure 1 ---
% ========================================================
figure(1) 
subplot(3, 1, 1); hold on;
for i = 1:N
    plot(t_span(idx), norm_E_tilde_i(i, idx));
end
ylabel('norm E_tilde_i');

subplot(3, 1, 2); hold on;
for i = 1:N
    plot(t_span(idx), norm_xi_i(i, idx));
end
ylabel('norm xi_i');

subplot(3, 1, 3); hold on;
for i = 1:N
    plot(t_span(idx), norm_eta_tilde_i(i, idx));
end
ylabel('norm eta_tilde_i'); 
xlabel('time/s');

% ========================================================
% --- Figure 2 ---
% ========================================================
figure(2) 
hold on;
for i = 1:N
    if B_topo(i,i) ~= 0
        plot(t_span(idx), squeeze(X_d_hat(i, N+1, idx))); 
    end
    for j = 1:N
        if A_topo(i,j) ~= 0
            plot(t_span(idx), squeeze(X_d_hat(i, j, idx))); 
        end
    end
end
ylabel('d_hat'); 
xlabel('time/s');

% ========================================================
% --- Figure 3 ---
% ========================================================
figure(3) 
hold on;
for i = 1:N
    plot(t_span(idx), X_Q_hat(i, idx)); 
end
ylabel('Q_hat'); 
xlabel('time/s');

% ========================================================
% --- Figure 4 ---
% ========================================================
figure(4) 
subplot(2, 2, 1); hold on;
plot(t_span(idx), X_v0(1, idx));
plot(t_span(idx), -X_v0(1, idx));
for i = 1:N
    plot(t_span(idx), reshape(X_q(1, idx, i), 1, []));
end
ylabel('q1');

subplot(2, 2, 2); hold on;
plot(t_span(idx), zeros(size(idx)));
for i = 1:N
    err_1 = reshape(X_q(1, idx, i), 1, []) - w_i(i) * X_v0(1, idx);
    plot(t_span(idx), err_1);
end
ylabel('e1');

subplot(2, 2, 3); hold on;
plot(t_span(idx), X_v0(2, idx));
plot(t_span(idx), -X_v0(2, idx));
for i = 1:N
    plot(t_span(idx), reshape(X_q(2, idx, i), 1, []));
end
ylabel('q2'); 
xlabel('time/s');

subplot(2, 2, 4); hold on;
plot(t_span(idx), zeros(size(idx)));
for i = 1:N
    err_2 = reshape(X_q(2, idx, i), 1, []) - w_i(i) * X_v0(2, idx);
    plot(t_span(idx), err_2);
end
ylabel('e2'); 
xlabel('time/s');

% ========================================================
% --- Figure 5 ---
% ========================================================
figure(5) 
X_dv0 = E_0 * X_v0;

subplot(2, 2, 1); hold on;
plot(t_span(idx), X_dv0(1, idx));
plot(t_span(idx), -X_dv0(1, idx));
for i = 1:N
    plot(t_span(idx), reshape(X_dq(1, idx, i), 1, []));
end
ylabel('dq1');

subplot(2, 2, 2); hold on;
plot(t_span(idx), zeros(size(idx)));
for i = 1:N
    err_dq_1 = reshape(X_dq(1, idx, i), 1, []) - w_i(i) * X_dv0(1, idx);
    plot(t_span(idx), err_dq_1);
end
ylabel('de1');

subplot(2, 2, 3); hold on;
plot(t_span(idx), X_dv0(2, idx));
plot(t_span(idx), -X_dv0(2, idx));
for i = 1:N
    plot(t_span(idx), reshape(X_dq(2, idx, i), 1, []));
end
ylabel('dq2'); 
xlabel('time/s');

subplot(2, 2, 4); hold on;
plot(t_span(idx), zeros(size(idx)));
for i = 1:N
    err_dq_2 = reshape(X_dq(2, idx, i), 1, []) - w_i(i) * X_dv0(2, idx);
    plot(t_span(idx), err_dq_2);
end
ylabel('de2'); 
xlabel('time/s');

% ========================================================
% --- Figure 6 ---
% ========================================================
figure(6) 
for m = 1:n
    subplot(2, 1, m); hold on;
    plot(t_span(idx), U_bound(idx)); 
    plot(t_span(idx), -U_bound(idx)); 
    for i = 1:N
        plot(t_span(idx), squeeze(X_z_err(m, idx, i)));
    end
    ylabel(['z_' num2str(m)]); 
    xlabel('time/s');
end

% ========================================================
% --- Figure 7 ---
% ========================================================
figure(7) 
hold on;
plot(t_span(idx), squeeze(X_u(1, idx, 1))); 
plot(t_span(idx), squeeze(X_u(2, idx, 1)));
ylabel('u1');
xlabel('time/s');