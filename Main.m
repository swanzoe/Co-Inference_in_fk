%% Code for simulations in manuscript
% <<Co-inference of Intra- and Inter-spectrum Prior Information for Underwater Array Signal Enhancement>>
clc
clear
close all
%% Parameters setting
c0=1500;                            % speed for waves
fs=1000;                            % sampling rate
T=5;                                % sampling time
dd=5;                               % sensor distance
M=50;                               % Sensor Number
d=dd*[0:1:M-1];
N =fs*T;                            % Snapshots
t=1/fs:1/fs:T;
S=2;                                % Target number
f0=[160 100];                       % target frequencies
wide_flag = [0;1];                  % wideband flag
width = 40;                         % band with
% Generate the distance and angles of sources
location_far=5000+6000*rand(S,1);
location=round(location_far);
theta_all=[45,30];
%% Multichannel time domain signal generation
signal=zeros(N,M);
s_wide = randn(N,1);                % generate wideband signals
bf = fir1(512,[0.2 0.25]);
S0 = filter(bf,1,s_wide);
S0_f = fft(S0);
f_plot_or = (1:N)/N*fs;
f_plot_or = f_plot_or';
figure;plot(f_plot_or,20*log10(abs(S0_f)));xlim([0 200])
for i=1:S
    r=location(i);
    theta=theta_all(i);
    dis = r;
    if wide_flag(i) == 0
        % Single frequency
        s = 50*sin( 2*pi * (f0(i) * (t'+d * sind(theta)/c0) )) ./ dis;
    else 
        % wide band
        A = 400*exp(2i*pi * f_plot_or * d * sind(theta)/c0)./ dis;
        s = ifft(A.*S0_f,'symmetric');
    end
    signal=signal+s;
end
% Add white noise
signal_n=awgn(signal,30);

%% Co-inference process
window = 1*fs;                      % window length
f=-window/2:window/2-1;             % frequency for fk spectrum
k=-M/2:M/2-1;                       % wavenumber for fk spectrum
f=f/window.*fs; 
phi = [];                           % save the phase matrix for each fk spectrum
FrameData = [];                     % save the data for each frame
for i = 1:round(N/window)
    receive = signal_n((i-1)*window+1:i*window,:);
    S1=receive;
    y=fft2(S1)/(window * M);
    yy=fftshift(y); 
    figure(1);
    mesh(k,f,abs(yy));ylim([50 200]);
    phi{i} = angle(yy);
    FrameData{i} = receive;
end
%% Co-inference process
N_frame = window;                       % number of pixels in each direction 
C = round(N/window);                    % Frame numbers
sigma_sq = sum(var(signal_n-signal));   % noise variance
rng('default'); rng(1,'twister'); 
% Free parameters of the methods 
reg_type = 'TV'; % regualrization operator for R: total variation
order = 1;
r = -1; gamma = 1; sigma = 10^(-3); % Free parameters

% Construct sampling operator for L1 problem: F and R matrix  
for c=1:C  
    m{c} = nnz(FrameData{c});
    opF{c} = @(x,mode) FkTransform(N_frame, M, x, mode); % F:linear forward operator
    noise_var{c} = sigma_sq; % we use the same noise variance for all frames, can be changed with known information 
end
% Construct function handle for the regularization operator
opR = @(x,flag) ReOperator( N_frame, M, x, reg_type, order, flag );

% The coference process with the sequence fk spectra
[DenoisedFk, ~, ~,~] = CoInference( C, opF, FrameData, noise_var, opR, r, gamma, sigma, 0 );
%% Covert back to multichannel signals
data_after = zeros(size(signal_n));
figure;
for c=1:C  
    fk_temp = abs(reshape(DenoisedFk{c},N_frame,M)); 
    fk_input = abs(fft2(FrameData{c}));   
    fk_after = max(fk_input(:))/max(fk_temp(:))*( fk_temp.*cos(phi{c}) + 1i*fk_temp.*sin(phi{c}));
    index = (c-1)*window+1:c*window;
    data_after(index,:) = ifft2(reshape(DenoisedFk{c},N_frame,M),'symmetric');
    subplot(1,2,1);
    mesh(k,f,fk_input);title('Input fk spectrum','fontsize',12);
    subplot(1,2,2);
    mesh(k,f,abs(fk_after));title('The denoised fk spectrum','fontsize',12)   
end 
%% Multichannel time-domain compare -- Fig.6
close all
figureHandle = tight_subplot(1,3,[.1 .06],[.15 .06],[.05 .05]);
color_before = [0,113,190]/255;
compare_color = [238,179,42]/255;
sensor_index = 6:10;

axes(figureHandle(1)); plot_multichan(t, signal_n(:,sensor_index));xlim([1.65 1.85]);
xlabel('Time (s)');title('(a) Multi-channel: Before')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);hold off

axes(figureHandle(2)); 
plot_multichan(t, data_after(:,sensor_index));xlim([1.65 1.85]);
xlabel('Time (s)');title('(b) Multi-channel: After')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);hold off

sensor_index = 6;
time_index = (1.65*fs+1:1.85*fs);
data_before = signal_n(time_index,sensor_index);
data_after_compare = data_after(time_index,sensor_index);
data_after_compare = data_after_compare/max(data_after_compare(:)) * max(data_before(:));
axes(figureHandle(3)); 
plot(t(time_index),data_before,'Color',color_before,'LineWidth',1);hold on;
plot(t(time_index),data_after_compare,'Color',compare_color,'LineWidth',2);
title('(c) Ch 01 Compare')
legend('Before','After');xlabel('Time (s)');ylabel('Amplitude');xlim([1.65 1.85]);ylim([-0.1 0.1]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14,'GridLineStyle','--');hold off
%% Compare in FFT and DAS ----Fig.7
color_before = [0,113,190]/255;
compare_color = [238,179,42]/255;
figure;
figureHandle = tight_subplot(1,2,[.06 .06],[.15 .1],[.15 .1]);

data1 = signal_n;
data2 = data_after/max(data_after(:)) * max(data1(:));
axes(figureHandle(1)); 
f_plot = (1:length(data1))/length(data1) *fs;
fft_data1 = 20*log10(abs(fft(data1(:,7))));
fft_data2 = 20*log10(abs(fft(data2(:,7))));
plot(f_plot,fft_data1,'Color',color_before,'LineWidth',1);hold on
plot(f_plot,fft_data2,'Color',compare_color,'LineWidth',1);
legend('Before','After');xlabel('Frequency(Hz)');ylabel('Power(dB)');
title('(a) FFT Spectrum');grid on;
xlim([0 200]); ylim([-20 50]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14,'GridLineStyle','--');hold off


axes(figureHandle(2)); 
d_kelm = d;
theta_range = -89:90;
[I_BF_before] = fun_das(data1,d_kelm',fs,theta_range,[100 200]);
[I_BF_after] = fun_das(data2,d_kelm',fs,theta_range,[100 200]);
I_BF_after_plot = I_BF_after/(max(I_BF_after(:))) * max(I_BF_before(:));
plot1 = 20*log10(I_BF_before);
plot2 = 20*log10(I_BF_after_plot);
plot(theta_range,plot1-max(plot1),'Color',color_before,'LineWidth',1);hold on
plot(theta_range,plot2-max(plot2),'Color',compare_color,'LineWidth',1);hold off
legend('Before','After');
title('(b) DAS Spectrum');xlabel('Theta Range');ylabel('Normalized Power(dB)')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14,'GridLineStyle','--');grid on;
xticks([-90:30:90]);ylim([-50 5])