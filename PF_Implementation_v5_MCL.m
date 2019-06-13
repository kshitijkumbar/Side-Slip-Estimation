clear all; clc; close all;

high_speed = false;
high_speed2 = false;

if(high_speed2)
    load('High_Speed2.mat');
    idx1 = 50000;
    idx2 = 60000;
elseif(high_speed)
    load('High_Speed.mat');
    idx1 = 10000;
    idx2 = 16000;
else
    load('Low_Speed.mat');
    idx1 = 12000;%8700;
    idx2 = 16000;%16700;
end

setup_niki2;


ax_mps2 = smooth(ax_mps2(idx1:idx2),10);
ay_mps2 = smooth(ay_mps2(idx1:idx2),10);
Fx_Commanded_N = Fx_Commanded_N(idx1:idx2);
Ux_mps = Ux_mps(idx1:idx2);
Uy_mps = Uy_mps(idx1:idx2);
delta_rad = delta_rad(idx1:idx2);
r_radps = r_radps(idx1:idx2);

delay = 0;
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

setup_niki;
%process and measurement noise
Q =0.1*diag([10 1 1])*dt^2;  %update this
R = 0.1*eye(3);     %update this
npf = 1000; % #particles
%% PF
clear X;
x_pf = zeros(3 , length ( t ) );
x_pf ( : , 1 ) = [5;0.75;0.59];
sig_pf( : , : , 1 ) = eye ( 3 ) ;
X( : , : , 1 ) = mvnrnd( x_pf ( : , 1 ) , sig_pf( : , : , 1 ) , npf )';
pf_weight( 1 , : ) = ones ( 1 , npf ) / npf ; %initial guess
tic ;
mixture_p = 0.55
for i = 2 : length( t )
     i;
    if Fx_Commanded_N(i) > 0
    Fxf = Fx_Commanded_N(i);
    Fxr = 0; 
    else
    Fxf = 0.64*Fx_Commanded_N(i);
    Fxr = 0.36*Fx_Commanded_N(i);
    end
% predict
y = [ax_mps2(i);ay_mps2(i-1);r_radps(i-1)]; %Measurement at the time step

for j = 1 : npf 
    if j<mixture_p*npf
    Xbar( : , j , i ) = f(X(:,j,i-1),delta_rad(i-1),Fxr,Fxf,dt,veh,tire_r,tire_f); 
    
%     [ X(1,j,i-1) + dt*(X(2,j,i-1)*X(3,j,i-1) + (Fxr + Fxf*cos(delta_rad(i-1)) - caf*sin(delta_rad(i-1))*(delta_rad(i-1) - (X(2,j,i-1) + a*X(3,j,i-1))/X(1,j,i-1)))/m);...
%                   X(2,j,i-1) - dt*(X(1,j,i-1)*X(3,j,i-1) - (Fxf*sin(delta_rad(i-1)) + caf*cos(delta_rad(i-1))*(delta_rad(i-1) - (X(2,j,i-1) + a*X(3,j,i-1))/X(1,j,i-1)) - (car*(X(2,j,i-1) - b*X(3,j,i-1)))/X(1,j,i-1))/m);...
%                   X(3,j,i-1) + (dt*(Fxf*a*sin(delta_rad(i-1)) + a*caf*cos(delta_rad(i-1))*(delta_rad(i-1) - (X(2,j,i-1) + a*X(3,j,i-1))/X(1,j,i-1)) + (b*car*(X(2,j,i-1) - b*X(3,j,i-1)))/X(1,j,i-1)))/Iz]+ mvnrnd(zeros(1,3),Q)'; 
    else
%          Xbar( : , j , i )=[X(1,j,i-1) + dt*(y(1) + y(3)*X(2,j,i-1));
%                             X(2,j,i-1) + dt*(y(2) - y(3)*X(1,j,i-1));
%                             y(3)]+ mvnrnd(zeros(1,3),R)';

           Xbar( : , j , i )=[x_pf(1,i-1) + dt*(y(1) + y(3)*x_pf(2,i-1));
                            x_pf(2,i-1) + dt*(y(2) - y(3)*x_pf(1,i-1));
                            y(3)]+ mvnrnd(zeros(1,3),R)';
    end
end
% update
for j = 1 : npf
    if j<mixture_p*npf
        y_est( : , j ) = g(Xbar( : , j , i ),delta_rad(i-1),Fxr,Fxf,dt,veh,tire_r,tire_f);
    else
        y_est( : , j ) = g(Xbar( : , j , i ),delta_rad(i-1),Fxr,Fxf,dt,veh,tire_r,tire_f);
        end
% [(Fxr + Fxf*cos(delta_rad(i-1)) + Caf*sin(delta_rad(i-1))*(delta_rad(i-1) - (uy + a*r)/ux))/m;
%                  (Fxf*sin(delta_rad(i-1)) + Caf*cos(delta_rad(i-1))*(delta_rad(i-1) - (uy + a*r)/ux) + (Car*(uy - b*r))/ux)/m;
%                   r ];
pf_weight( i , j ) = mvnpdf ( y ,y_est( : , j ), R) ;
pf_weight_c( i , j ) = pf_weight( i , j );
end

pf_weight(i,:) = pf_weight(i,:)/sum(pf_weight(i,:)) ; % normalize weight
x_pf( : , i ) = wmean(pf_weight( i , : ) ,Xbar ( : , : , i ) ) ;
sig_pf( : , : , i ) = wcov( pf_weight( i , : ) ,Xbar ( : , : , i ) , Xbar ( : , : , i ) ) ;
X( : , : , i ) = Xbar ( : , : , i ) ;
% importace resampling
if mod( i , 5) == 0
% if abs(ay_mps2(i-1))<1
indices = randsample ( npf , npf , true , pf_weight( i , : ) ) ;
X( : , : , i ) = X( : , indices , i ) ;
pf_weight( i , : ) = ones ( 1 , npf ) / npf; 
end

end
t_PF = toc ;
% plot(t,Uy_mps,t,x_pf(2,:))

figure()

subplot(3,1,1)

plot(t, x_pf(1,:))
title("PARTICLE FILTER");
hold on;
plot(t,Ux_mps(1:end));
ylabel("U_x")
legend("Estimate","Truth");

subplot(3,1,2)
plot(t, x_pf(2,:))
hold on;
plot(t,Uy_mps(1:end));
ylabel("U_y")
ylim([-2 5]);
legend("Estimate","Truth");


subplot(3,1,3)
plot(t, x_pf(3,:))
hold on;
plot(t,r_radps(1:end));
ylabel("r")
ylim([-1 3]);
legend("Estimate","Truth");

xlabel("Time [s]");

figure()

for i= 1:length(Ux_mps)
    beta_estimate(i) = calculate_beta(x_pf(2,i),x_pf(1,i));
    beta_truth(i) = calculate_beta(Uy_mps(i),Ux_mps(i));
end

beta_mae = norm(beta_estimate - beta_truth);
plot(t,beta_estimate,t,beta_truth);
title("Sideslip Angle Estimation Performance: RMS Error = " + beta_mae);
ylabel("Sideslip Angle \beta");
xlabel("Time [s]");
ylim([-0.2,0.2]);
legend("Estimate","Truth");

ux_mae = norm(Ux_mps - x_pf(1,:))
uy_mae = norm(Uy_mps - x_pf(2,:))
r_mae = norm(r_radps - x_pf(3,:))
beta_mae/length(beta_mae)
%%
function wc = wcov ( weight , x , y )
% x ( : , i ) y ( : , i ) r e p r e s e n t samples
n = size(x , 2 ) ; % number o f samples
mux = wmean( weight , x ) ;
muy = wmean( weight , y ) ;
wc = ( x - mux * ones ( 1 , n) ) * diag ( weight ) *( y - muy * ones ( 1 , n) )';
end

function wm = wmean( weight , x )
% weight i s row ve c t o r
% x ( : , i ) r e p r e s e n t s samples
%r eut run column ve c t o r
wm = sum( x * weight',2) ;
end

%Calculate Side Slip Angle
function beta = calculate_beta(Uy,Ux)
   
   if( (1000*norm(Uy) < norm(Ux)) || norm(Ux) < 1  )
       beta = 0;
   else
     beta = atan(Uy/Ux);
   end
end

function [mu_eff_f, mu_eff_r] = effective_friction_coeff(Fzof,Fzif,Fzor,Fzir)

    mu_of = 1.25 - 0.00004*Fzof;
    mu_if = 1.25 - 0.00004*Fzif;
    mu_or = 1.25 - 0.00004*Fzor;
    mu_ir = 1.25 - 0.00004*Fzir;
    
    mu_eff_f = (mu_of*Fzof + mu_if*Fzif)/(Fzof+Fzif);
    mu_eff_r = (mu_or*Fzor + mu_ir*Fzir)/(Fzor + Fzir);

end

function [Fzof, Fzif, Fzor, Fzir, phi] = weight_transfer(ax,ay,veh)
    
    Kf = 70000; Kr = 50000; grav= 9.81;
    Kphi = Kf + Kr;
    
    h1 = veh.hcg - (veh.b/veh.L*veh.hf + veh.a/veh.L*veh.hr);
    phi = veh.m*grav*h1*ay/grav/(Kphi - veh.m*grav*h1);
    
    deltaFzf_lat = 1/veh.tf*(Kf*phi + veh.Wf*veh.hf*ay/grav);
    deltaFzr_lat = 1/veh.tr*(Kr*phi + veh.Wr*veh.hr*ay/grav);
        
    deltaFzf_long = -veh.m*ax*veh.hcg/veh.L;
    
    Fzof = veh.Wf/2 + deltaFzf_long/2 + deltaFzf_lat;
    Fzif = veh.Wf/2 + deltaFzf_long/2 - deltaFzf_lat;
    
    Fzor = veh.Wr/2 - deltaFzf_long/2 + deltaFzr_lat;
    Fzir = veh.Wr/2 - deltaFzf_long/2 - deltaFzr_lat;
    

end




function [Fy, frac] = coupled_model(alpha, tire, Fx,mu,Fz)
    
    ca = tire.Ca;
    
       
    if( (mu*Fz)^2 - Fx^2 < 0 )
        zeta = 0;
        
    else
        zeta = sqrt((mu*Fz)^2 - Fx^2)/mu/abs(Fz);
    end
    
    alpha_sl = abs(atan2(3*zeta*mu*Fz,ca));
 
    if(abs(alpha)<alpha_sl)
        Fy = -ca*tan(alpha) + ca^2/(3*zeta*mu*Fz)*abs(tan(alpha))*tan(alpha)...
            - ca^3/(27*zeta^2*mu^2*Fz^2)*(tan(alpha))^3;
    else
        Fy = -zeta*mu*Fz*sign(alpha);
    end
    frac = norm([Fx,Fy])/norm(mu*Fz);
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
    
    % lateral tire forces
  
    persistent ay;
    persistent ax;
    if(isempty(ay))
        ay = 0;
        ax = (Fxr+Fxf)/veh.m;
    end
     [Fzof, Fzif, Fzor, Fzir, phi] = weight_transfer(ax,ay,veh);
   [mu_eff_f, mu_eff_r] = effective_friction_coeff(Fzof,Fzif,Fzor,Fzir);
    
    if(Fzof<0 || Fzif <0 || Fzor <0 || Fzir <0)
      
    end
            
     [Fyf, fracf] = coupled_model(alpha_f, tire_f, Fxf,mu_eff_f,Fzof+Fzif);
     [Fyr, fracr] = coupled_model(alpha_r, tire_r, Fxr,mu_eff_r,Fzor+Fzir);
     
%      Fyf = linear_model(alpha_f,tire_f,Fxf);
%      Fyr = linear_model(alpha_r,tire_r,Fxr);
       
    xnext = [Ux;Uy;r] +  dt*[r*Uy + (Fxr + Fxf - Fyf*delta - 0.5*rho*C_DA*Ux^2 - frr*m*9.81)/m;...
    -r*Ux + (Fyf + Fyr + Fxf*delta)/m;...
    (a*Fyf + a*Fxf*delta - b*Fyr)/Iz] ;
 
    ax = r*Uy + (Fxr + Fxf - Fyf*delta - 0.5*rho*C_DA*Ux^2 - frr*m*9.81)/m;
    ay =  -r*Ux + (Fyf + Fyr + Fxf*delta)/m;
  
    
end


function y = g(x,delta,Fxr,Fxf,dt,veh,tire_r,tire_f)
       
    Ux = x(1);    Uy = x(2);    r = x(3);    a = veh.a;    b = veh.b;    m = veh.m;  Iz = veh.Iz;
    
    [alpha_f, alpha_r] = slip_angles( r, Uy, Ux, delta, veh);
    
    % lateral tire forces
    %ax = Fx/veh.m;
    persistent ay;
    persistent ax;
    if(isempty(ay))
        ay = 0;
        ax = (Fxr+Fxf)/veh.m;
    end
     [Fzof, Fzif, Fzor, Fzir, phi] = weight_transfer(ax,ay,veh);
   [mu_eff_f, mu_eff_r] = effective_friction_coeff(Fzof,Fzif,Fzor,Fzir);
    
%     if(Fzof<0 || Fzif <0 || Fzor <0 || Fzir <0)
% %         "ERROR"
%     end
            
%      [Fyf, fracf] = coupled_model(alpha_f, tire_f, Fxf,mu_eff_f,Fzof+Fzif);
%      [Fyr, fracr] = coupled_model(alpha_r, tire_r, Fxr,mu_eff_r,Fzor+Fzir);
%         
          Fyf = linear_model(alpha_f,tire_f,Fxf);
     Fyr = linear_model(alpha_r,tire_r,Fxr);
         
    y =  [(Fxr + Fxf - Fyf*delta)/m;...
    (Fyf + Fyr + Fxf*delta)/m;...
    r];

    ax = y(1);
    ay = y(2);    
      
end