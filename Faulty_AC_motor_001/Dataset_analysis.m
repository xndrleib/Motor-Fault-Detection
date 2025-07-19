%% Script which reads .csv files, unifies their data for the curent 
% separately for phases A, B, C.
clear all
close all
clc

datasetfolder = 'C:\WorkFolder\Projects\Motor-Fault-Detection\dataset\engine_2\experiment_1\current';
% assign the pathes
all_SOHs = dir(datasetfolder);
% Create dirname_nofault and dirname_shortcircuit with paths to the folders
% with experiment results without fault and with short circuit 
for dir_idx=1:length(all_SOHs)
    if strcmp(all_SOHs(dir_idx).name,'2nd_load_100')
        dirname_nofault = [all_SOHs(dir_idx).folder '\' all_SOHs(dir_idx).name];
    elseif strcmp(all_SOHs(dir_idx).name,'4th_load_100')
        dirname_shortcircuit = [all_SOHs(dir_idx).folder '\' all_SOHs(dir_idx).name];
    end
end
%{
% Create variables with data for different phases separately
for phasenum=1:3
    all_files_ph_nofault = dir([dirname_nofault '\' num2str(phasenum)]);
    for filenum = 1:length(all_files_ph_nofault)
        if strfind(all_files_ph_nofault(filenum).name,'.csv')
            file = all_files_ph_nofault(filenum).name;
            path = all_files_ph_nofault(filenum).folder;
            opts = detectImportOptions([path '\' file]);
            opts.DataLines = [2,Inf];
            expdata = readmatrix([path '\' file],opts);
            figure
            plot(expdata(:,2),expdata(:,3))
            title(['nofault_phase' num2str(phasenum) '_' file],'Interpreter', 'none');
        end
    end
end
for phasenum=1:3
    all_files_ph_shortcircuit = dir([dirname_shortcircuit '\' num2str(phasenum)]);
    for filenum = 1:length(all_files_ph_shortcircuit)
        if strfind(all_files_ph_shortcircuit(filenum).name,'.csv')
            file = all_files_ph_shortcircuit(filenum).name;
            path = all_files_ph_shortcircuit(filenum).folder;
            opts = detectImportOptions([path '\' file]);
            opts.DataLines = [2,Inf];
            expdata = readmatrix([path '\' file],opts);
            figure
            plot(expdata(:,2),expdata(:,3))
            title(['short circuit_phase' num2str(phasenum) '_' file],'Interpreter', 'none');
        end
    end
end
close all
% if strcmp(all_SOHs(dir_idx).name,'2nd_load_100')
%}
%% Save data to .csv files
%no fault
input_AIR80B4_2;
idx = find(out.I_out_curr.time >= 1, 1, 'first');
data_time_nf = out.I_out_curr.time(idx:end)-1;
data_current_nf_phA = out.I_out_curr.signals.values(idx:end,1);
data_current_nf_phB = out.I_out_curr.signals.values(idx:end,2);
data_current_nf_phC = out.I_out_curr.signals.values(idx:end,3);
figure
plot(data_time_nf,data_current_nf_phA)
title('nofault_phase_A_model','Interpreter','none');
M1_nf_1 = [(1:length(data_time_nf))', data_time_nf(:), data_current_nf_phA];
writematrix(M1_nf_1, '2_nofault_load100_phaseA.csv');
figure
plot(data_time_nf,data_current_nf_phB)
title('nofault_phase_B_model','Interpreter','none');
M1_nf_2 = [(1:length(data_time_nf))', data_time_nf, data_current_nf_phB];
writematrix(M1_nf_2, '2_nofault_load100_phaseB.csv');
figure
plot(data_time_nf,data_current_nf_phC)
title('nofault_phase_C_model','Interpreter','none');
M1_nf_3 = [(1:length(data_time_nf))', data_time_nf, data_current_nf_phC];
writematrix(M1_nf_3, '2_nofault_load100_phaseC.csv');

% short circuit
input_AIR80B4_2_sc;
idx = find(out.I_out_curr.time >= 1, 1, 'first');
data_time_sc = out.I_out_curr.time(idx:end)-1;
data_current_sc_phA = out.I_out_curr.signals.values(idx:end,1);
data_current_sc_phB = out.I_out_curr.signals.values(idx:end,2);
data_current_sc_phC = out.I_out_curr.signals.values(idx:end,3);
figure
plot(data_time_sc,data_current_sc_phA)
title('shortcircuit_phase_A_model','Interpreter','none');
M1_sc_1 = [(1:length(data_time_sc))', data_time_sc(:), data_current_sc_phA];
writematrix(M1_sc_1, '2_shortcircuit_load100_phaseA.csv');
figure
plot(data_time_sc,data_current_sc_phB)
title('shortcircuit_phase_B_model','Interpreter','none');
M1_sc_2 = [(1:length(data_time_sc))', data_time_sc(:), data_current_sc_phB];
writematrix(M1_sc_2, '2_shortcircuit_load100_phaseB.csv');
figure
plot(data_time_sc,data_current_sc_phC)
title('shortcircuit_phase_C_model','Interpreter','none');
M1_sc_3 = [(1:length(data_time_sc))', data_time_sc(:), data_current_sc_phC];
writematrix(M1_sc_3, '2_shortcircuit_load100_phaseC.csv');
