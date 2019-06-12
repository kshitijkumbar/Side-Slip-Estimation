% EKF Fiala model implementation

clc;
clear all;
close all;

%% Load Data
high_speed = true;
low_speed = false;

if(low_speed == true)
    
    start = 12000;
    stop = 16000;
    load('Low_Speed.mat');

else if(high_speed == true)
    
    start = 10000;
    stop = 16000;
    load('High_Speed.mat');
    end
end

ax_mps2 = MovingSmooth(ax_mps2(start:stop), 13);
ay_mps2 = MovingSmooth(ay_mps2(start:stop),13);
r_radps = r_radps(start:stop);
Ux_mps = Ux_mps(start:stop);
Uy_mps = Uy_mps(start:stop);
delta_rad = delta_rad(start:stop);
Fx_Commanded_N = Fx_Commanded_N(start:stop);

g = 9.81;                       % [m/s^2]  gravity

%% vehicle parameters
veh.m  = 1926.2;                % [kg]     mass
veh.Iz = 2763.49;               % [kg-m^2] rotational inertia
veh.a  = 1.264;                 % [m]      distance from CoM to front axle
veh.b  = 1.367;                 % [m]      distance from C0M to rear axle
veh.L  = veh.a + veh.b;         % [m]      wheelbase
veh.Wf = veh.m*(veh.b/veh.L);   % [kg]     front axle weight
veh.Wr = veh.m*(veh.a/veh.L);   % [kg]     rear axle weight
veh.rW = 0.318;                 % [m]      tire radius

%% Tire Parameters
% front tires
tire_f.Ca_lin =  80000;         % [N/rad]  linear model cornering stiffness
tire_f.Ca     = 110000;         % [N/rad]  fiala model cornering stiffness
tire_f.mu_s   = 0.90;           %          sliding friction coefficient
tire_f.mu     = 0.90;           %          peak friction coefficient
tire_f.Fz     = veh.Wf*g;       % [N]      static normal load on the axle
% rear tires
tire_r.Ca_lin = 120000;
tire_r.Ca     = 180000;
tire_r.mu_s   = 0.94;
tire_r.mu     = 0.94;
tire_r.Fz     = veh.Wr*g;

%% Setting Up Implementation Details

dt = 0.005;
t = 0:dt:(length(Ux_mps)-1)*dt;
n = length(t);
Fxr = zeros(1,n);
Fxf = zeros(1,n);
delta = delta_rad;
Q = 0.008*diag([10 1 1])*dt^2;
R = 0.03*eye(3);
y_meas = [ax_mps2; ay_mps2; r_radps];

if(low_speed == true)
    mu0 = [5.0852;0.7717;0.5870];
else
    mu0 = [16.5891; -0.4705; 0.5503];
end
sigma0 = 0.001*eye(3);
mu_update = zeros(3,n);
mu_predict = zeros(3,n);

%% Implementation

%Getting the force control inputs

tic;

for i=1:n
    
    if(Fx_Commanded_N(i) > 0)
        Fxf(i) = Fx_Commanded_N(i);
        Fxr(i) = 0;
    
    else
        
        Fxf(i) = 0.64*Fx_Commanded_N(i);
        Fxr(i) = 0.36*Fx_Commanded_N(i);
        
    end
end

% Implementing EKF

for i=1:n-1
    
    % First predict step
    if(i == 1)
        
        % Computing alpha_f and alpha_r
        [alpha_f, alpha_r] = Gen_alpha(mu0, veh, delta(i));
        % Computing Fyf and Fyr
        [Fyf, Fyr, c, d] = Fiala_Model(alpha_f, alpha_r, tire_f, tire_r);
        % Getting the Dynamics model
        F = Dynamics_Model(mu0, Fxr(i), Fxf(i), delta(i), Fyr, Fyf, veh);
        % Getting the Jacobian
        Jac_A = Jacobian_A(mu0, veh, tire_f, tire_r, delta(i), dt, c, d);
        
        % The prediction step
        mu_predict(:,i) = mu0 + F*dt;
        cov_predict(:,:,i) = (Jac_A)*sigma0*(Jac_A)' + Q;
        
    else
        
        % All other predict steps
        
        % Computing alpha_f and alpha_r
        [alpha_f, alpha_r] = Gen_alpha(mu_update(:,i-1), veh, delta(i));
        % Computing Fyf and Fyr
        [Fyf, Fyr, c, d] = Fiala_Model(alpha_f, alpha_r, tire_f, tire_r);
        % Getting the Dynamics model
        F = Dynamics_Model(mu_update(:,i-1), Fxr(i), Fxf(i), delta(i), Fyr, Fyf, veh);
        % Getting the Jacobian
        Jac_A = Jacobian_A(mu_update(:,i-1), veh, tire_f, tire_r, delta(i), dt, c, d);
        
        % The prediction step
        mu_predict(:,i) = mu_update(:, i-1) + F*dt;
        cov_predict(:,:,i) = (Jac_A)*cov_update(:,:,i-1)*(Jac_A)' + Q;
        
    end
    
    % update step
    
    % Computing new alpha_f and alpha_r
   [alpha_f, alpha_r] = Gen_alpha(mu_predict(:,i), veh, delta(i+1));
   % Computing Fyf and Fyr
   [Fyf, Fyr, c, d] = Fiala_Model(alpha_f, alpha_r, tire_f, tire_r);
   % Getting the Dynamics model at the update step
    F = Dynamics_Model(mu_predict(:,i), Fxr(i+1), Fxf(i+1), delta(i+1), Fyr, Fyf, veh);
    % Getting the Jacobian at the update step
    Jac_A = Jacobian_A(mu_predict(:,i), veh, tire_f, tire_r, delta(i+1), dt, c, d);
    %Getting the Jacobian of the measurement model
    Jac_C = Jacobian_C(mu_predict(:,i), veh, tire_f, tire_r, delta(i+1), dt, c, d);
    % Getting the predicted measurement
    y_hat(:,i) = y_predict(mu_predict(:,i), Fxr(i+1), Fxf(i+1), Fyf, Fyr, delta(i+1), veh);
    % The update step
    mu_update(:,i) = mu_predict(:,i) + cov_predict(:,:,i)*((Jac_C)')*inv(((Jac_C)*cov_predict(:,:,i)*(Jac_C)' + R))*(y_meas(:,i+1) - y_hat(:,i));
    cov_update(:,:,i) = cov_predict(:,:,i) - cov_predict(:,:,i)*(Jac_C)'*inv((Jac_C)*cov_predict(:,:,i)*(Jac_C)' + R)*(Jac_C)*cov_predict(:,:,i);
    %Det_cov = det(cov_update(:,:,i))
    
end

toc;
% Getting the computation time
Total_Comp_Time = toc;
Per_Iter_Time = toc/n;

%% Plotting

% Plotting for low speed data
if (low_speed == true)

figure(1);
subplot(2,1,1)
plot(t, Ux_mps, 'r');
hold on
plot(t, mu_update(1,:),'b');
grid on
ylabel('U_x (m/s)');
ylim([0 13]);
xlim([0.5 19.5]);
title('EKF for Non-Linear Tire Model');
xlabel('Time (s)');
legend('True','Estimate');
hold off
subplot(2,1,2)
plot(t, mu_update(2,:),'b');
hold on
grid on
ylabel('U_y (m/s)');
ylim([-0.2 0.9]);
xlim([0.5 19.5]);
plot(t, Uy_mps,'r');
legend('Estimate','True');
xlabel('Time (s)');
hold off


else

% Plotting for high speed data
figure(1);
subplot(2,1,1)
plot(t, Ux_mps, 'r');
hold on
plot(t, mu_update(1,:),'b');
grid on
ylabel('U_x (m/s)');
%ylim([0 13]);
xlim([0.5 29.5]);
title('EKF for Non-Linear Tire Model');
xlabel('Time (s)');
legend('True','Estimate');
hold off
subplot(2,1,2)
plot(t, mu_update(2,:),'b');
hold on
grid on
ylabel('U_y (m/s)');
%ylim([-0.1 0.9]);
xlim([0.5 29.5]);
plot(t, Uy_mps,'r');
legend('Estimate','True');
xlabel('Time (s)');
hold off
end

figure(2);

for i= 1:length(Ux_mps)
    beta_estimate(i) = calculate_beta(mu_update(2,i),mu_update(1,i));
    beta_truth(i) = calculate_beta(Uy_mps(i),Ux_mps(i));
end

beta_mae_norm = norm(beta_estimate - beta_truth)/length(beta_estimate)*1000;
plot(t,beta_estimate,t,beta_truth);
title("Sideslip Angle Estimation Performance: RMS Error = " + beta_mae_norm);
ylabel("Sideslip Angle \beta");
xlabel("Time [s]");
ylim([-0.2,0.2]);
grid on
legend("Estimate","Truth");

%% Helper Functions

%Calculate Side Slip Angle
function beta = calculate_beta(Uy,Ux)
   
   if( (1000*norm(Uy) < norm(Ux)) || norm(Ux) < 1  )
       beta = 0;
   else
     beta = atan(Uy/Ux);
   end
end


function [alpha_f, alpha_r] = Gen_alpha(mu, veh, delta)

% Computing the alpha_f and alpha_r values
alpha_f = atan2((mu(2) + veh.a*mu(3)),mu(1)) - delta;
alpha_r = atan2((mu(2) - veh.b*mu(3)),mu(1));

end

function[Fyf, Fyr, c, d] = Fiala_Model(alpha_f, alpha_r, tire_f, tire_r)

% Computing the Fyf and Fyr values
    Fzf = tire_f.Fz;
   alpha_slf = abs(3*tire_f.mu*Fzf/tire_f.Ca); 
    Fzr = tire_r.Fz;
   alpha_slr = abs(3*tire_r.mu*Fzr/tire_r.Ca); 
   
    if(abs(alpha_f)<alpha_slf)
        Fyf = -tire_f.Ca*tan(alpha_f) + tire_f.Ca^2/(3*tire_f.mu*Fzf)*(2-tire_f.mu_s/tire_f.mu)*abs(tan(alpha_f))*tan(alpha_f)...
            - tire_f.Ca^3/(9*tire_f.mu^2*Fzf^2)*(tan(alpha_f))^3*(1-2*tire_f.mu_s/(3*tire_f.mu));
        c = 1;
    else
        Fyf = -tire_f.mu_s*Fzf*sign(alpha_f);
        c = 0;
    end
    
     if(abs(alpha_r)<alpha_slr)
        Fyr = -tire_r.Ca*tan(alpha_r) + tire_r.Ca^2/(3*tire_r.mu*Fzr)*(2-tire_r.mu_s/tire_r.mu)*abs(tan(alpha_r))*tan(alpha_r)...
            - tire_r.Ca^3/(9*tire_r.mu^2*Fzr^2)*(tan(alpha_r))^3*(1-2*tire_r.mu_s/(3*tire_r.mu));
        d = 1;
    else
        Fyr = -tire_r.mu_s*Fzr*sign(alpha_r);
        d = 0;
    end
    
end


function F = Dynamics_Model(mu, Fxr, Fxf, delta, Fyr, Fyf, veh)

a = veh.a;
b = veh.b;
m = veh.m;
Iz = veh.Iz;

% Computing the Dynamics model

F = [(mu(3)*mu(2) + (Fxr + Fxf*cos(delta) - Fyf*sin(delta))/m); ...
     (-mu(3)*mu(1) + (Fyr + Fyf*cos(delta) + Fxf*sin(delta))/m); ...
     (a*Fyf*cos(delta) + a*Fxf*cos(delta) - b*Fyr)/Iz];
 
end

function Jac_A = Jacobian_A(mu, veh, tire_f, tire_r, delta, dt, c, d)

ux = mu(1);
uy = mu(2);
r = mu(3);
Caf = tire_f.Ca_lin;
Car = tire_r.Ca_lin;
a = veh.a;  b = veh.b; m = veh.m;  Iz = veh.Iz;  Fzf = tire_f.Fz;  Fzr = tire_r.Fz;
mu_fs = tire_f.mu_s;    muf = tire_f.mu;   mu_rs = tire_r.mu_s;    mur = tire_r.mu;
  

% Getting the Jacobian for the Dynamics model

if(c == 1 && d == 1)  
    
Jac_A = [  1 - (dt*sin(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)))/m,                                                                                                                                                                                                                                                                                                dt*(r + (sin(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)))/m),                                                                                                                                                                                                                                                               dt*(uy + (sin(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)))/m);
           -dt*(r - (cos(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)) + (Car*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1))/ux^2 + (Car^2*abs(tan((uy - b*r)/ux))*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2))/m),    1 - (dt*(cos(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) + (Car*(tan((uy - b*r)/ux)^2 + 1))/ux + (Car^2*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux)))/m,      -dt*(ux - ((Car*b*(tan((uy - b*r)/ux)^2 + 1))/ux - cos(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) + (Car^2*b*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*b*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux))/m);
         -(dt*(b*((Car*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1))/ux^2 + (Car^2*abs(tan((uy - b*r)/ux))*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2)) - a*cos(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2))))/Iz,  (dt*(b*((Car*(tan((uy - b*r)/ux)^2 + 1))/ux + (Car^2*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux)) - a*cos(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux))))/Iz,    1 - (dt*(b*((Car*b*(tan((uy - b*r)/ux)^2 + 1))/ux + (Car^2*b*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*b*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux)) + a*cos(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux))))/Iz];

end

if(c == 0 && d == 1)
        
Jac_A = [  1 - (2*Fzf*dt*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta)*(uy + a*r))/(m*ux^2),                                                                                                                                                                                                                                                                                                    dt*(r + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta))/(m*ux)),                                                                                                                                                                                                                                                               dt*(uy + (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta))/(m*ux));
          -dt*(r - ((Car*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1))/ux^2 + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta)*(uy + a*r))/ux^2 + (Car^2*abs(tan((uy - b*r)/ux))*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2))/m),    1 - (dt*((Car*(tan((uy - b*r)/ux)^2 + 1))/ux + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux + (Car^2*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux)))/m,      -dt*(ux - ((Car*b*(tan((uy - b*r)/ux)^2 + 1))/ux - (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux + (Car^2*b*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*b*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux))/m);
         -(dt*(b*((Car*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1))/ux^2 + (Car^2*abs(tan((uy - b*r)/ux))*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2)) - (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta)*(uy + a*r))/ux^2))/Iz, (dt*(b*((Car*(tan((uy - b*r)/ux)^2 + 1))/ux + (Car^2*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux)) - (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux))/Iz,      1 - (dt*(b*((Car*b*(tan((uy - b*r)/ux)^2 + 1))/ux + (Car^2*b*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*b*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux)) + (2*Fzf*a^2*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux))/Iz];

end

if(c == 1 && d == 0)
        
Jac_A = [ 1 - (dt*sin(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)))/m,                                                         dt*(r + (sin(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)))/m),                                               dt*(uy + (sin(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)))/m);
         -dt*(r - (cos(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)) + (2*Fzr*mu_rs*dirac((uy - b*r)/ux)*(uy - b*r))/ux^2)/m),   1 - (dt*(cos(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) + (2*Fzr*mu_rs*dirac((uy - b*r)/ux))/ux))/m,    -dt*(ux + (cos(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) - (2*Fzr*b*mu_rs*dirac((uy - b*r)/ux))/ux)/m);
         (dt*(a*cos(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)) - (2*Fzr*b*mu_rs*dirac((uy - b*r)/ux)*(uy - b*r))/ux^2))/Iz, -(dt*(a*cos(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) - (2*Fzr*b*mu_rs*dirac((uy - b*r)/ux))/ux))/Iz,   1 - (dt*(a*cos(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) + (2*Fzr*b^2*mu_rs*dirac((uy - b*r)/ux))/ux))/Iz];

end
if(c == 0 && d == 0)

Jac_A = [  1 - (2*Fzf*dt*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta)*(uy + a*r))/(m*ux^2),                                                            dt*(r + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta))/(m*ux)),                                               dt*(uy + (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta))/(m*ux));
          -dt*(r - ((2*Fzr*mu_rs*dirac((uy - b*r)/ux)*(uy - b*r))/ux^2 + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta)*(uy + a*r))/ux^2)/m),     1 - (dt*((2*Fzr*mu_rs*dirac((uy - b*r)/ux))/ux + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux))/m,    -dt*(ux - ((2*Fzr*b*mu_rs*dirac((uy - b*r)/ux))/ux - (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux)/m);
         -(dt*((2*Fzr*b*mu_rs*dirac((uy - b*r)/ux)*(uy - b*r))/ux^2 - (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta)*(uy + a*r))/ux^2))/Iz,     (dt*((2*Fzr*b*mu_rs*dirac((uy - b*r)/ux))/ux - (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux))/Iz,     1 - (dt*((2*Fzr*b^2*mu_rs*dirac((uy - b*r)/ux))/ux + (2*Fzf*a^2*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux))/Iz];
end

end

function Jac_C = Jacobian_C(mu, veh, tire_f, tire_r, delta, dt, c, d)

ux = mu(1);
uy = mu(2);
r = mu(3);
Caf = tire_f.Ca_lin;
Car = tire_r.Ca_lin;
a = veh.a;  b = veh.b; m = veh.m;  Iz = veh.Iz;  Fzf = tire_f.Fz;  Fzr = tire_r.Fz;
mu_fs = tire_f.mu_s;    muf = tire_f.mu;   mu_rs = tire_r.mu_s;    mur = tire_r.mu;

% Getting the Jacobian for the measurement model

if(c == 1 && d == 1)
   
Jac_C = [                                                                                                                                                                                                                                                                                         -(sin(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)))/m,                                                                                                                                                                                                                                                    (sin(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)))/m,                                                                                                                                                                                                                                                         (sin(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)))/m;
         (cos(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)) + (Car*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1))/ux^2 + (Car^2*abs(tan((uy - b*r)/ux))*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2))/m, -(cos(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) + (Car*(tan((uy - b*r)/ux)^2 + 1))/ux + (Car^2*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux))/m, ((Car*b*(tan((uy - b*r)/ux)^2 + 1))/ux - cos(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) + (Car^2*b*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*b*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux))/m;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     1];
 
end   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
if(c == 0 && d == 1)
    
Jac_C = [                                                                                                                                                                                                                                                                                         -(2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta)*(uy + a*r))/(m*ux^2),                                                                                                                                                                                                                                      -(2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta)*(uy + a*r))/(m*ux^2),                                                                                                                                                                                                                                                         (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta))/(m*ux);
         ((Car*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1))/ux^2 + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta)*(uy + a*r))/ux^2 + (Car^2*abs(tan((uy - b*r)/ux))*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(uy - b*r)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux^2))/m,    -((Car*(tan((uy - b*r)/ux)^2 + 1))/ux + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux + (Car^2*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux))/m,   ((Car*b*(tan((uy - b*r)/ux)^2 + 1))/ux - (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux + (Car^2*b*abs(tan((uy - b*r)/ux))*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux) + (Car^2*b*sign(tan((uy - b*r)/ux))*tan((uy - b*r)/ux)*(tan((uy - b*r)/ux)^2 + 1)*(mu_rs/mur - 2))/(3*Fzf*muf*ux))/m;
                                                                                                                                                                                                                                                                                                                                                                          0,                                                                                                                                                                                                                                                                                                               0,                                                                                                                                                                                                                                                                                                                      1];
end 
if(c == 1 && d == 0)

Jac_C = [                                                     -(sin(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)))/m,                                          (sin(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)))/m,                                            (sin(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)))/m;
         (cos(delta)*((Caf*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux^2 - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux^2) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(uy + a*r)*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux^2)) + (2*Fzr*mu_rs*dirac((uy - b*r)/ux)*(uy - b*r))/ux^2)/m, -(cos(delta)*((Caf*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) + (2*Fzr*mu_rs*dirac((uy - b*r)/ux))/ux)/m, -(cos(delta)*((Caf*a*(tan(delta - (uy + a*r)/ux)^2 + 1))/ux - (Caf^3*a*tan(delta - (uy + a*r)/ux)^2*(tan(delta - (uy + a*r)/ux)^2 + 1)*((2*mu_fs)/(3*muf) - 1))/(3*Fzf^2*muf^2*ux) + (Caf^2*a*abs(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux) + (Caf^2*a*tan(delta - (uy + a*r)/ux)*sign(tan(delta - (uy + a*r)/ux))*(tan(delta - (uy + a*r)/ux)^2 + 1)*(mu_fs/muf - 2))/(3*Fzf*muf*ux)) - (2*Fzr*b*mu_rs*dirac((uy - b*r)/ux))/ux)/m;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        1];
     
end
if(c == 0 && d == 0)
    
Jac_C = [                                                     -(2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta)*(uy + a*r))/(m*ux^2),                                          (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta))/(m*ux),                                           (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*sin(delta))/(m*ux);
         ((2*Fzr*mu_rs*dirac((uy - b*r)/ux)*(uy - b*r))/ux^2 + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta)*(uy + a*r))/ux^2)/m, -((2*Fzr*mu_rs*dirac((uy - b*r)/ux))/ux + (2*Fzf*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux)/m, ((2*Fzr*b*mu_rs*dirac((uy - b*r)/ux))/ux - (2*Fzf*a*mu_fs*dirac(delta - (uy + a*r)/ux)*cos(delta))/ux)/m;
                                                                                                                                      0,                                                                                                     0,                                                                                                        1];
 
       
end                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
end

function y_hat = y_predict(mu, Fxr, Fxf, Fyf, Fyr, delta, veh)

u_x = mu(1);
u_y = mu(2);
r = mu(3);
m = veh.m;

% Getting the predicted measurement

y_hat = [ (Fxr + Fxf*cos(delta) - Fyf*sin(delta))/m ; ...
          (Fyr + Fyf*cos(delta) + Fxf*sin(delta))/m ; ...
          r];
      
end

function Avg = MovingSmooth(Data, N)

% Function that performs a moving average on measurements

Avg = zeros(1, length(Data));
start_index = (N-1)/2;
stop_index = length(Data) - start_index;
window = (N-1)/2;
Sum = 0;
for i = 1:length(Data)
    
    if(i <= start_index)
        Avg(i) = Data(i);
    
    else if(i > stop_index)
            Avg(i) = Data(i);
            
    else
            for j = 1:window
                Sum = Sum + Data(i-j);
                Sum = Sum + Data(i+j);
                Avg(i) = (Sum + Data(i))/N;
            end
            Sum = 0;
        end
    end
end
end

