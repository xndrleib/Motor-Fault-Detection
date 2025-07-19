%% Parameters of the motor AIR 80B4
%
Vdc = 460; % DC Voltage in the drive
Ts = 2e-5; % invertor time constant
P = 2; % number of pole pairs
Power = 2*746; % W - motor power (1.5 kW)
Vph = 380/sqrt(3); % V - phase voltage
Vm = sqrt(2)*Vph; % V - magnetizing voltage
fb = 50; % Hz - base frequency
% we = 2*pi*fb;
wb = 2*pi*fb;          % rpm - Base speed
we = 1000*2*pi/60; % rad/s - reference rotation frequency
Rr = 1.274;             % Ohm - Rotor resistance
Rs = 3.523;             % Ohm - Stator resistance
Lls = 0.013;            % H - Stator inducatnce
Llr = 0.013;            % H - Rotor inductance

% specific magnetic loading (average flux density in air gap, in Tesla or Wb/m²)
Bmax = 0.01; % Bmax - maximum flux per pole (Wb)
Kcdistf = 0.9; % Distribution factor
Kpitchf = 1; % Pitch factor
kw = Kcdistf*Kpitchf; % kw - winding factor (accounts for coil distribution and pitch)
nturns_phase = round(Vph*sqrt(3)/4.44/fb/kw/Bmax); % total number of winding turns
%% Faulty mode. Short circuit of the stator winding
%fscnturns = 10; % number of turns of the stator winding wherre there is a short circuit
%Rs = Rs*((nturns_phase-fscnturns)/nturns_phase);
%Rr = Rr*((nturns_phase-fscnturns)/nturns_phase)^2;
%Lls = Lls*((nturns_phase-fscnturns)/nturns_phase)^2;
%Llr = Llr*((nturns_phase-fscnturns)/nturns_phase)^2;

%% Calculation of other parameters

Lm = 0.401;            % H - Magnetizing Inductance
J = 0.0226;           % kg/m^2 - Moment of inertia
% Impedance and angular speed calculations

Xls     = wb*Lls;           % Ohm - Stator impedance
Xlr     = wb*Llr;           % Ohm - Rotor impedance
Xm      = wb*Lm;            % Ohm - Magnetizing impedance
Xml  = 1/(1/Xls+1/Xm+1/Xlr); % Ohm - Xm with stator and rotor leakage reactances

%Tlim=11; % Nm - maximal torque developed by the motor

% input 0 - 100
%Kp=10;
%Ki=10;
%Ktorquegain = 1;
%Tlim=11;

% input 300
Tlim=11;
Kp=10;
Ki=10;
Ktorquegain = 1;
fc=3800;
F=0.001;
%}

Tsimulation = 4000e-3+1000e-3; %s - total simulation time
Tswrite = 0.24e-3; %414e-3; %s - data saving frequency

out = sim('AC_motor_2_torque_control');