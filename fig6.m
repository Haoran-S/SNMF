% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% MATLAB code for our BSUM algorithm, to reproduce our works on SNMF research.
% Simply run fig6.m, you will get the result for figure 6 shown in section V (C)
% To get results for other figures, slightly modification may apply.
% 
% References:
% [1] Qingjiang Shi, Haoran Sun, Songtao Lu, Mingyi Hong, and Meisam Razaviyayn.
% "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization."
% arXiv preprint arXiv:1607.03092 (2016).
% 
% version 1.0 -- April/2016
% Written by Haoran Sun (hrsun AT iastate.edu)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

clear
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subject to change
n = 100;     % number of data points
r = 10;       % number of clusters
maxtime = 5; % max running time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nMC = 1;
maxIter = 100000;
nf=r;
Xdata = exprnd(1, [n nf]);
M = generate_sparse_correlation_kernel(Xdata, .5, 0.1);
% M = generate_self_tuning_gaussian_kernel(Xdata);
for loop=1:nMC
    loop
    
    disp('running ANLS algorithm:')
    [H0, H, obj_vec_BCD, grad_BCD, time_BCD] = symnmf_anls(M, r, 1, maxIter, maxtime);
    P1 =sqrt(obj_vec_BCD);
    P1_grad = grad_BCD;
    P1_time = time_BCD;
    
    disp('running Newton algorithm:')
    [X2, iter, obj, obj_vec_Newton, grad_Newton, time_Newton] = symnmf_newton(M, H0, r, 1, maxIter, maxtime);
    P2=sqrt(obj_vec_Newton);
    P2_grad = grad_Newton;
    P2_time = time_Newton;
    
    disp('running rEVD algorithm:')
    [obj_vec_rEVD X3 grad_rEVD, time_rEVD] = uniqsymnmf(M, H0, r, maxIter, maxtime);
    P3=sqrt(obj_vec_rEVD);
    P3_grad = grad_rEVD;
    P3_time = time_rEVD;
    
    disp('running cyclic sBSUM algorithm:')
    [X4 obj_vec_BSUM grad_BSUM time_BSUM] = SNMF_cyclic_BSUM(M, maxIter, H0', maxtime);
    P4=sqrt(obj_vec_BSUM);
    P4_grad = grad_BSUM;
    P4_time = time_BSUM;
    
    disp('running cyclic vBSUM algorithm:')
    [X5 obj_vec_vBSUM grad_vBSUM, time_vBSUM] = SNMF_cyclic_vBSUM(M, maxIter, H0, maxtime);
    P5=sqrt(obj_vec_vBSUM);
    P5_grad = grad_vBSUM;
    P5_time = time_vBSUM;
    
end

time1 = mean(P1_time, 1);
time2 = mean(P2_time, 1);
time3 = mean(P3_time, 1);
time4 = mean(P4_time, 1);
time5 = mean(P5_time, 1);

avg1 = mean(P1,1);
avg2 = mean(P2,1);
avg3 = mean(P3,1);
avg4 = mean(P4,1);
avg5 = mean(P5,1);

scale = norm(M, 'fro')/100;

figure(1)
clf
set(gcf, 'color', 'white')
plot(time1, avg1/scale, '-.k');
hold on;
plot(time2, avg2/scale, '.c');
hold on;
plot(time3, avg3/scale, ':r');
hold on;
plot(time4, avg4/scale, 'b--');
hold on;
plot(time5, avg5/scale, 'm-');
ylabel('100 ||M-XX^T|| / ||M||')
xlabel('Cpu time (s)')
legend('ANLS','Newton', 'rEVD', 'sBSUM', 'vBSUM');
savefig('fig6a.fig');

avg1_grad = mean(P1_grad,1);
avg2_grad = mean(P2_grad,1);
avg3_grad = mean(P3_grad,1);
avg4_grad = mean(P4_grad,1);
avg5_grad = mean(P5_grad,1);

figure(2)
clf
set(gcf, 'color', 'white')
semilogy(time1, avg1_grad, '-.k');
hold on;
semilogy(time2, avg2_grad, '.c');
hold on;
semilogy(time3, avg3_grad, ':r');
hold on;
semilogy(time4, avg4_grad, 'b--');
hold on;
semilogy(time5, avg5_grad, 'm-');
ylabel('Optimality gap')
xlabel('Cpu time (s)')
legend('ANLS','Newton', 'rEVD', 'sBSUM', 'vBSUM');
savefig('fig6b.fig');


