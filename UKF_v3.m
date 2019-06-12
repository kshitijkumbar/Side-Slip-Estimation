%% UKF

clear all; clc; close all;
    
high_speed = false;
high_speed2 = false;

if(high_speed2)
    load('High_Speed2.mat');
    idx1 = 75000;
    idx2 = 85000;
elseif(high_speed)
    load('High_Speed.mat');
    idx1 = 10000;
    idx2 = 16000;
else
    load('Low_Speed.mat');
    idx1 = 12000;%8700;
    idx2 = 16000;%16700;
end

setup_niki;


ax_mps2 = smooth(ax_mps2(idx1:idx2),10);
ay_mps2 = smooth(ay_mps2(idx1:idx2),10);
Fx_Commanded_N = Fx_Commanded_N(idx1:idx2);
Ux_mps = Ux_mps(idx1:idx2);
Uy_mps = Uy_mps(idx1:idx2);
delta_rad = delta_rad(idx1:idx2);
r_radps = r_radps(idx1:idx2);

delay = 40;
Fx_Commanded_N = Fx_Commanded_N(delay+1:end);
Fx_Commanded_N = [zeros(1,delay) Fx_Commanded_N];

%time 
dt = 0.005;
t = 0:dt:(length(Uy_mps)-1)*dt;

figure()
subplot(4,1,1)
plot(t,ax_mps2)
grid on;
ylabel("a_x");
subplot(4,1,2)
plot(t,Fx_Commanded_N);grid on;
ylabel("F_x");
subplot(4,1,3)
plot(t,Ux_mps);grid on;
ylabel("U_x");
subplot(4,1,4)
plot(t,Uy_mps);grid on;
ylabel("U_y");

% subplot(7,1,1)
% plot(t,ax_mps2)
% ylabel("a_x");
% subplot(7,1,2)
% plot(t,Fx_Commanded_N)
% ylabel("F_x");
% subplot(7,1,3)
% plot(t,Ux_mps)
% ylabel("U_x");
% subplot(7,1,4)
% plot(t,ay_mps2)
% ylabel("a_y");
% subplot(7,1,5)
% plot(t,delta_rad)
% ylabel("\delta");
% subplot(7,1,6)
% plot(t,r_radps)
% ylabel("r");
% subplot(7,1,7)
% plot(t,Uy_mps)
% ylabel("U_y");

%process and measurement noise
Q = 0.1*diag([10 1 1])*dt^2;  %update this
R = 0.01*eye(3);     %update this


%% Dynamics

d0 = [Ux_mps(1);Uy_mps(1);r_radps(1)];

d_store = [d0];

d = d0;

for i = 1:length(t)
    %generate control input
    if(Fx_Commanded_N(i) > 0)
        Fxf = Fx_Commanded_N(i);
        Fxr = 0;
    else
        Fxf = 0.64*Fx_Commanded_N(i);
        Fxr = 0.36*Fx_Commanded_N(i);
    end    
    delta = delta_rad(i);
         
    d = f(d,delta,Fxr,Fxf,dt,veh,tire_r,tire_f);
        
    
    d_store = [d_store d];
end

figure()


subplot(3,1,1)
plot(t, d_store(1,1:end-1),t,Ux_mps)
ylabel("U_x")
legend("Dynamics","Truth");

title("Simulation of the dynamics");

subplot(3,1,2)
plot(t, d_store(2,1:end-1),t,Uy_mps)
ylabel("U_y")
legend("Dynamics","Truth");

subplot(3,1,3)
plot(t, d_store(3,1:end-1),t,r_radps)
ylabel("r")
legend("Dynamics","Truth");

%% UKF

mu0 = [Ux_mps(1);Uy_mps(1);r_radps(1)];
sigma0 = 0.01*eye(3);

mu_store = [mu0];
sigma_store = [sigma0];

mu = mu0;
sigma = sigma0;

lambda = 2;
n = length(mu);
num_points = 2*n+1;

tic;

for i = 1:length(t)-1
    
       
    %PREDICT
    
    %generate control input
    if(Fx_Commanded_N(i) > 0)
        Fxf = Fx_Commanded_N(i);
        Fxr = 0;
    else
        Fxf = 0.64*Fx_Commanded_N(i);
        Fxr = 0.36*Fx_Commanded_N(i);
    end    
    delta = delta_rad(i);
    
    
    [x w] = UT(mu,sigma,lambda,num_points); 
    
    %noise free propagation
    for j = 1:num_points
        x(:,j) = f(x(:,j),delta,Fxr,Fxf,dt,veh,tire_r,tire_f);
    end   
    
    %fit to gaussian
    [mu sigma] = UTINV(x,w,num_points);    
    sigma = sigma + Q;   
    
    for k =1:1 %iterative UKF
            [x w] = UT(mu,sigma,lambda,num_points);  

            for j = 1:num_points
                y(:,j) = g(x(:,j),delta,Fxr,Fxf,dt,veh,tire_r,tire_f);
            end     
            %expected measurement
            yhat = sum(y'.*w')';

            sigmay = 0;   sigmaxy = 0;  sigmayx = 0;   
            for j =1:num_points
                sigmay = sigmay +  w(j)*(y(:,j) - yhat)*(y(:,j) - yhat)';
                sigmaxy = sigmaxy + w(j)*(x(:,j) - mu)*(y(:,j) - yhat)';
                sigmayx = sigmayx + w(j)*(y(:,j) - yhat)*(x(:,j) - mu)';
            end
            sigmay = sigmay + R;

            y_meas = [ax_mps2(i+1);ay_mps2(i+1);r_radps(i+1)];

            mu = mu + sigmaxy*inv(sigmay)*(y_meas-yhat);
            sigma = sigma - sigmaxy*inv(sigmay)*sigmayx;
    end
            mu_store = [mu_store mu];
    
%     mu
%     sigma
end

TIME = toc/(length(t)-1);

figure()

subplot(3,1,1)

plot(t, mu_store(1,1:end))
title("UKF");
hold on;
plot(t,Ux_mps(1:end));
ylabel("U_x")
legend("Estimate","Truth");

subplot(3,1,2)
plot(t, mu_store(2,1:end))
hold on;
plot(t,Uy_mps(1:end));
ylabel("U_y")
ylim([-2 5]);
legend("Estimate","Truth");


subplot(3,1,3)
plot(t, mu_store(3,1:end))
hold on;
plot(t,r_radps(1:end));
ylabel("r")
ylim([-1 3]);
legend("Estimate","Truth");

xlabel("Time [s]");

figure()

for i= 1:length(Ux_mps)
    beta_estimate(i) = calculate_beta(mu_store(2,i),mu_store(1,i));
    beta_truth(i) = calculate_beta(Uy_mps(i),Ux_mps(i));
end

beta_mae_norm = norm(beta_estimate - beta_truth)/length(beta_estimate)*1000;
plot(t,beta_estimate,t,beta_truth);
title("Sideslip Angle Estimation Performance: RMS Error = " + beta_mae_norm);
ylabel("Sideslip Angle \beta");
xlabel("Time [s]");
ylim([-0.2,0.2]);
legend("Estimate","Truth");

ux_mae_norm = norm(Ux_mps - mu_store(1,:))/length(Ux_mps)
uy_mae_norm = norm(Uy_mps - mu_store(2,:))/length(Uy_mps)
r_mae_norm = norm(r_radps - mu_store(3,:))/length(r_radps)
beta_mae_norm

%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%

%Calculate Side Slip Angle
function beta = calculate_beta(Uy,Ux)
   
   if( (1000*norm(Uy) < norm(Ux)) || norm(Ux) < 1  )
       beta = 0;
   else
     beta = atan(Uy/Ux);
   end
end

function Fy = coupled_model(alpha, tire, Fx)
     
    Fz = tire.Fz;    
    mu = tire.mu;
    ca = tire.Ca;
    
    if( (mu*Fz)^2 - Fx^2 < 0  )
        zeta = 0;
    else 
        zeta = sqrt((mu*Fz)^2 - Fx^2)/mu/Fz;
    end
    
   alpha_sl = abs(atan2(3*zeta*mu*Fz,ca));
 
    if(abs(alpha)<alpha_sl)
        Fy = -ca*tan(alpha) + ca^2/(3*zeta*mu*Fz)*abs(tan(alpha))*tan(alpha)...
            - ca^3/(27*zeta^2*mu^2*Fz^2)*(tan(alpha))^3;
    else
        Fy = -zeta*mu*Fz*sign(alpha);
    end
end


%Calculate Forces with the Fiala Nonlinear Tire Model
function Fy = fiala_model(alpha, tire,Fx)

   Fz = tire.Fz;
   alpha_sl = abs(3*tire.mu*Fz/tire.Ca); 
    if(abs(alpha)<alpha_sl)
        Fy = -tire.Ca*tan(alpha) + tire.Ca^2/(3*tire.mu*Fz)*(2-tire.mu_s/tire.mu)*abs(tan(alpha))*tan(alpha)...
            - tire.Ca^3/(9*tire.mu^2*Fz^2)*(tan(alpha))^3*(1-2*tire.mu_s/(3*tire.mu));
    else
        Fy = -tire.mu_s*Fz*sign(alpha);
    end
    
end

function Fy = linear_model(alpha,tire,Fx)  

       Fy = -tire.Ca_lin*alpha;
       
end

function [alpha_f, alpha_r] = slip_angles( r, Uy, Ux, delta, veh)

    alpha_f = atan2(Uy+veh.a*r, Ux) - delta ;
    alpha_r = atan2(Uy-veh.b*r,Ux);
    
end

function xnext = f(x,delta,Fxr,Fxf,dt,veh,tire_r,tire_f) 

    Ux = x(1);    Uy = x(2);    r = x(3);    a = veh.a;    b = veh.b;    m = veh.m;  Iz = veh.Iz;
    frr = 0.015 ; %rolling friction constant
    C_DA = 0.594 ; %coefficient of drag X surface area
    rho = 1.225; %density of air

    
    [alpha_f, alpha_r] = slip_angles( r, Uy, Ux, delta, veh);
    
    Fyf =   linear_model(alpha_f,tire_f,Fxf);
    Fyr =   linear_model(alpha_r,tire_r,Fxr);
    
    xnext = [Ux;Uy;r] +  dt*[r*Uy + (Fxr + Fxf - Fyf*delta - 0.5*rho*C_DA*Ux^2 - frr*m*9.81)/m;...
    -r*Ux + (Fyf + Fyr + Fxf*delta)/m;...
    (a*Fyf + a*Fxf*delta - b*Fyr)/Iz] ;
 
end


function y = g(x,delta,Fxr,Fxf,dt,veh,tire_r,tire_f)
       
    Ux = x(1);    Uy = x(2);    r = x(3);    a = veh.a;    b = veh.b;    m = veh.m;  Iz = veh.Iz;
    
    [alpha_f, alpha_r] = slip_angles( r, Uy, Ux, delta, veh);
    
    Fyf =   linear_model(alpha_f,tire_f,Fxf);
    Fyr =   linear_model(alpha_r,tire_r,Fxr);
    
    y =  [(Fxr + Fxf - Fyf*delta)/m;...
    (Fyf + Fyr + Fxf*delta)/m;...
    r];
      
end

function [x w] = UT(mu,sigma,lambda,num_points)
    n = length(mu);
    x = zeros(n,num_points);
    x(:,1) = mu;
    w(1) = lambda/(lambda+n);
%     sigchol = sqrtm(sigma);
    [u s v] = svd(sigma);
    sigchol = u*sqrtm(s);
    
    for i=2:n+1
        x(:,i) = mu + sqrt(n+lambda)*sigchol(:,i-1);
        w(i) =  1/2/(n+lambda);
    end
    for i=n+2:2*n+1
        x(:,i) = mu - sqrt(n+lambda)*sigchol(:,i-n-1);
        w(i) =  1/2/(n+lambda);
    end  
    
     
    
 end

function [mu sigma] = UTINV(x,w,num_points)
    n = length(x(:,1));
    mu = 0;
    sigma = zeros(n);
    
    for i=1:num_points
        mu = mu + w(i)*x(:,i);      
    end
    for i=1:num_points    
        sigma = sigma + w(i)*(x(:,i) - mu)*(x(:,i) - mu)';
    end
end