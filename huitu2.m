clear; close all; clc;


if exist('TrajectoryAndErrorData.mat', 'file') && ...
   exist('TrajectoryAndErrorData_BLF.mat', 'file') && ...
   exist('TrajectoryAndErrorData_GPPF.mat', 'file')
    
    data_pd = load('TrajectoryAndErrorData.mat');
    data_blf = load('TrajectoryAndErrorData_BLF.mat');
    data_gppf = load('TrajectoryAndErrorData_GPPF.mat');
else
    error('未找到必要的 .mat 数据文件，请确保三个文件都在当前目录下。');
end


fig_pos1 = [100, 450, 560, 420]; 
fig_pos2 = [100,  50, 560, 420]; 


figure(1);
set(gcf, 'Position', fig_pos1);
hold on; box on; grid on;

plot(data_pd.time_data, data_pd.leader_trajectory, 'k--', 'LineWidth', 1.5);
plot(data_pd.time_data, data_pd.agent_trajectory, 'b-', 'LineWidth', 1.2);
plot(data_blf.time_data, data_blf.agent_trajectory, 'g-', 'LineWidth', 1.2); 
plot(data_gppf.time_data, data_gppf.agent_trajectory, 'r-', 'LineWidth', 1.2); 

xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Position (rad)', 'Interpreter', 'latex');
title('Tracking Trajectory Comparison', 'Interpreter', 'latex');
legend('Leader $v_1$', 'Agent (PD)', 'Agent (BLF)', 'Agent (GPPF)', 'Location', 'best', 'Interpreter', 'latex');

figure(2);
set(gcf, 'Position', fig_pos2);
hold on; box on; grid on;

plot([0, max(data_pd.time_data)], [0, 0], 'k:', 'LineWidth', 1.5);

plot(data_pd.time_data, data_pd.tracking_error, 'b-', 'LineWidth', 1.2); 
plot(data_blf.time_data, data_blf.tracking_error, 'g-', 'LineWidth', 1.2); 
plot(data_gppf.time_data, data_gppf.tracking_error, 'r-', 'LineWidth', 1.2); 

xlabel('Time (s)', 'Interpreter', 'latex'); 
ylabel('Tracking Error $z_1$', 'Interpreter', 'latex');
title('Tracking Error Comparison', 'Interpreter', 'latex');
legend('Zero Ref', 'Error (PD)', 'Error (BLF)', 'Error (GPPF)', 'Location', 'best', 'Interpreter', 'latex');
