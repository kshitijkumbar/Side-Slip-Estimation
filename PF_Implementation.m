clear all; clc; close all;

high_speed = false;
high_speed2 = false;

if(high_speed2)
    load('High_Speed2.mat');
    idx1 = 50000;
    idx2 = 60000;
elseif(high_speed)
    load('High_Speed.mat');
    idx1 = 3000;
    idx2 = 8000;
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
% setup_niki;

a  = 1.264;                 
b  = 1.367; 
Caf = 80000;
Car = 120000;
caf = Caf;
car = Car;
Iz = 2763.49;
m  = 1926.2;
work_idx = 4001;


%process and measurement noise
Q = 0.1*diag([10 1 1])*dt^2;  %update this
R = 1*eye(3);     %update this

%% Particle Filter
%Initial Sample
mixture_p = 0.55
init_mean = [5;0.7;0.5];
init_cov = 0.1*eye(3);
samples = 100;
tic
for i =1:samples
    x(:,i,1) = mvnrnd(init_mean,init_cov);
end

for i = 1 : length(t)   %Time
    i
    if Fx_Commanded_N(i) > 0
    Fxf = Fx_Commanded_N(i);
    Fxr = 0; 
    else
    Fxf = Fx_Commanded_N(i)/2;
    Fxr = Fx_Commanded_N(i)/2;
    end
    delta = delta_rad(i);
    
     y_true = [ax_mps2(i);ay_mps2(i);r_radps(i)];
    for j = 1:samples   %Samples
    %Predict Step
     if j<mixture_p*sample
    noise_pred = mvnrnd(init_mean,Q);
    x_pred(:,j)=[ x(1,j,i) + dt*(x(2,j,i)*x(3,j,i) + (Fxr + Fxf*cos(delta) - caf*sin(delta)*(delta - (x(2,j,i) + a*x(3,j,i))/x(1,j,i)))/m);...
                  x(2,j,i) - dt*(x(1,j,i)*x(3,j,i) - (Fxf*sin(delta) + caf*cos(delta)*(delta - (x(2,j,i) + a*x(3,j,i))/x(1,j,i)) - (car*(x(2,j,i) - b*x(3,j,i)))/x(1,j,i))/m);...
                  x(3,j,i) + (dt*(Fxf*a*sin(delta) + a*caf*cos(delta)*(delta - (x(2,j,i) + a*x(3,j,i))/x(1,j,i)) + (b*car*(x(2,j,i) - b*x(3,j,i)))/x(1,j,i)))/Iz];
              else
              Xbar( : , j , i )=[x_pf(1,i-1) + dt*(y(1) + y(3)*x_pf(2,i-1));
                            x_pf(2,i-1) + dt*(y(2) - y(3)*x_pf(1,i-1));
                            y(3)]+ mvnrnd(zeros(1,3),R)';
    end
    
%     [x(1,j,i) + dt*nu*cos(x(3,j,i));
%                   x(2,j,i) + dt*nu*sin(x(3,j,i));
%                   x(3,j,i) + dt*sin(t(i))] + noise_pred';
%     %Update
                ux = x_pred(1,j);
                uy = x_pred(2,j);
                r = x_pred(3,j);
                ax = (Fxr + Fxf*cos(delta) + Caf*sin(delta)*(delta - (uy + a*r)/ux))/m;
                ay = (Fxf*sin(delta) + Caf*cos(delta)*(delta - (uy + a*r)/ux) - (Car*(uy - b*r))/ux)/m;
                r = r;
                y_hat = [ax;ay;r];
               
%               y_hat = 
%               y_hat = norm([ x_pred(1,j)  x_pred(2,j)]);
              weights_likelihood(j) = mvnpdf((y_true-y_hat),[0;0;0],R);% + mvnrnd(0,R);
    end
    
    weights_normalized = weights_likelihood./sum(weights_likelihood);
    if i< length(t)
    x(:,:,i+1) = [randsample(x_pred(1,:), samples, true, weights_likelihood);
                  randsample(x_pred(2,:), samples, true, weights_likelihood);
                  randsample(x_pred(3,:), samples, true, weights_likelihood)];
    end
end
Particle_time = toc
%%
for i =1 : length(t(1:work_idx))
Px_particle(i) = mean(x(1,:,i));
Py_particle(i) = mean(x(2,:,i));
theta_particle(i) = mean(x(3,:,i));
end
figure(4)
subplot(3,1,1)
plot(t(1:work_idx),Ux_mps(1:work_idx),t(1:work_idx),Px_particle)
xlabel('time')
ylabel('U_x')
legend('dynamic simulation','particle filter')   
subplot(3,1,2)
plot(t(1:work_idx),Uy_mps(1:work_idx),t(1:work_idx),Py_particle)
xlabel('time')
ylabel('U_y')
legend('dynamic simulation','particle filter')   
subplot(3,1,3)
plot(t(1:work_idx),r_radps(1:work_idx),t(1:work_idx),theta_particle)
xlabel('time')
ylabel('r')
legend('dynamic simulation','particle filter')    
figure(5)
plot(t(1:work_idx),Uy_mps(1:work_idx)./Ux_mps(1:work_idx),t(1:work_idx),Py_particle./Px_particle)

%

%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%

%Calculate Forces with the Fiala Nonlinear Tire Model
function Fy = fiala_model(alpha, tire)
%   Calculate tire forces with the fiala model

%%%%% STUDENT CODE HERE %%%%%

    Fz = tire.Fz;
   alpha_sl = abs(3*tire.mu*Fz/tire.Ca);
 
    if(abs(alpha)<alpha_sl)
        Fy = -tire.Ca*tan(alpha) + tire.Ca^2/(3*tire.mu*Fz)*(2-tire.mu_s/tire.mu)*abs(tan(alpha))*tan(alpha)...
            - tire.Ca^3/(9*tire.mu^2*Fz^2)*(tan(alpha))^3*(1-2*tire.mu_s/(3*tire.mu));
    else
        Fy = -tire.mu_s*Fz*sign(alpha);
    end
%%%%% END STUDENT CODE %%%%%
end



function xnext = f(x,delta,Fxr,Fxf,dt,veh,tire_r,tire_f)
    
    Ux = x(1);    Uy = x(2);    r = x(3);    a = veh.a;    b = veh.b;    m = veh.m;
    car = tire_r.Ca_lin; caf = tire_f.Ca_lin; Iz = veh.Iz;

    xnext =[      Ux + dt*(Uy*r + (Fxr + Fxf*cos(delta) - caf*sin(delta)*(delta - (Uy + a*r)/Ux))/m);...
 Uy - dt*(Ux*r - (Fxf*sin(delta) + caf*cos(delta)*(delta - (Uy + a*r)/Ux) - (car*(Uy - b*r))/Ux)/m);...
  r + (dt*(Fxf*a*sin(delta) + a*caf*cos(delta)*(delta - (Uy + a*r)/Ux) + (b*car*(Uy - b*r))/Ux))/Iz];
 
end


function y = g(x,delta,Fxr,Fxf,dt,veh,tire_r,tire_f)
    
    Ux = x(1);    Uy = x(2);    r = x(3);    a = veh.a;    b = veh.b;    m = veh.m;
    car = tire_r.Ca_lin; caf = tire_f.Ca_lin; Iz = veh.Iz;
    
    y = [(Fxr + Fxf*cos(delta) - caf*sin(delta)*(delta - (Uy + a*r)/Ux))/m;...
 (Fxf*sin(delta) + caf*cos(delta)*(delta - (Uy + a*r)/Ux) - (car*(Uy - b*r))/Ux)/m;...
      r];
      
end