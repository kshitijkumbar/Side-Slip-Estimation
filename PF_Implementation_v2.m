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
Fx_Commanded_N = 0.8*Fx_Commanded_N(delay+1:end);
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
R = 0.01*eye(3);     %update this
npf = 5000; % #particles
%% PF
clear X;
x_pf = zeros(3 , length ( t ) );
x_pf ( : , 1 ) = [5;0.75;0.59];
sig_pf( : , : , 1 ) = eye ( 3 ) ;
X( : , : , 1 ) = mvnrnd( x_pf ( : , 1 ) , sig_pf( : , : , 1 ) , npf )';
pf_weight( 1 , : ) = ones ( 1 , npf ) / npf ; %initial guess
tic ;
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
for j = 1 : npf 
    Xbar( : , j , i ) = [ X(1,j,i-1) + dt*(X(2,j,i-1)*X(3,j,i-1) + (Fxr + Fxf*cos(delta_rad(i-1)) - caf*sin(delta_rad(i-1))*(delta_rad(i-1) - (X(2,j,i-1) + a*X(3,j,i-1))/X(1,j,i-1)))/m);...
                  X(2,j,i-1) - dt*(X(1,j,i-1)*X(3,j,i-1) - (Fxf*sin(delta_rad(i-1)) + caf*cos(delta_rad(i-1))*(delta_rad(i-1) - (X(2,j,i-1) + a*X(3,j,i-1))/X(1,j,i-1)) - (car*(X(2,j,i-1) - b*X(3,j,i-1)))/X(1,j,i-1))/m);...
                  X(3,j,i-1) + (dt*(Fxf*a*sin(delta_rad(i-1)) + a*caf*cos(delta_rad(i-1))*(delta_rad(i-1) - (X(2,j,i-1) + a*X(3,j,i-1))/X(1,j,i-1)) + (b*car*(X(2,j,i-1) - b*X(3,j,i-1)))/X(1,j,i-1)))/Iz] + mvnrnd(zeros(1,3),Q)'; 
end
y = [ax_mps2(i-1);ay_mps2(i-1);r_radps(i-1)]; %Measurement at the time step
% update
for j = 1 : npf
                ux = Xbar( 1 , j , i );
                uy = Xbar( 2 , j , i );
                r = Xbar( 3 , j , i );
y_est( : , j ) =[(Fxr + Fxf*cos(delta_rad(i-1)) + Caf*sin(delta_rad(i-1))*(delta_rad(i-1) - (uy + a*r)/ux))/m;
                 (Fxf*sin(delta_rad(i-1)) + Caf*cos(delta_rad(i-1))*(delta_rad(i-1) - (uy + a*r)/ux) + (Car*(uy - b*r))/ux)/m;
                  r ];%+ mvnrnd(zeros(1,3),R)';
pf_weight( i , j ) = mvnpdf ( y ,y_est( : , j ), R);
pf_weight_c( i , j ) = pf_weight( i , j );
end

pf_weight(i,:) = pf_weight(i,:)/sum(pf_weight(i,:)) ; % normalize weight
x_pf( : , i ) = wmean(pf_weight( i , : ) ,Xbar ( : , : , i ) ) ;
sig_pf( : , : , i ) = wcov( pf_weight( i , : ) ,Xbar ( : , : , i ) , Xbar ( : , : , i ) ) ;
X( : , : , i ) = Xbar ( : , : , i ) ;
% importace resampling
if mod( i , 1 ) == 0
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
function beta = calculate_beta(Uy,Ux)
   
   if( (1000*norm(Uy) < norm(Ux)) || norm(Ux) < 1  )
       beta = 0;
   else
     beta = atan(Uy/Ux);
   end
end


