clear; clc;

try
    data_pd = load('TrajectoryAndErrorData.mat');
    data_blf = load('TrajectoryAndErrorData_BLF.mat');
    data_gppf = load('TrajectoryAndErrorData_GPPF.mat');
catch
    error('未找到数据文件，请确保三个 .mat 文件都在当前工作目录下！');
end


e_pd = data_pd.tracking_error(:);
e_blf = data_blf.tracking_error(:);
e_gppf = data_gppf.tracking_error(:);

t = data_pd.time_data(:);
dt = mean(diff(t)); 


ISE_pd  = sum(e_pd.^2) * dt;        
IAE_pd  = sum(abs(e_pd)) * dt;     
Mean_pd = mean(abs(e_pd));         

ISE_blf  = sum(e_blf.^2) * dt;
IAE_blf  = sum(abs(e_blf)) * dt;
Mean_blf = mean(abs(e_blf));

ISE_gppf  = sum(e_gppf.^2) * dt;
IAE_gppf  = sum(abs(e_gppf)) * dt;
Mean_gppf = mean(abs(e_gppf));

fprintf('====================================================\n');
fprintf('             控制策略误差性能指标对比\n');
fprintf('====================================================\n\n');

fprintf('【1. 一般 PD 控制】\n');
fprintf('   ISE  (积分平方误差)   : %.6f\n', ISE_pd);
fprintf('   IAE  (积分绝对误差)   : %.6f\n', IAE_pd);
fprintf('   MAE  (平均绝对误差)   : %.6f\n\n', Mean_pd);

fprintf('【2. 传统 BLF 控制】\n');
fprintf('   ISE  (积分平方误差)   : %.6f\n', ISE_blf);
fprintf('   IAE  (积分绝对误差)   : %.6f\n', IAE_blf);
fprintf('   MAE  (平均绝对误差)   : %.6f\n\n', Mean_blf);

fprintf('【3. 所提 GPPF 控制】\n');
fprintf('   ISE  (积分平方误差)   : %.6f\n', ISE_gppf);
fprintf('   IAE  (积分绝对误差)   : %.6f\n', IAE_gppf);
fprintf('   MAE  (平均绝对误差)   : %.6f\n\n', Mean_gppf);
