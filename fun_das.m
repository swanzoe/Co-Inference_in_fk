function [I_BF] = fun_das(X,loc,fs,theta,f_range)
c = 1500;
N_fft = length(X);
f_axis =(0:N_fft-1)/N_fft*fs;
S = fft(X, N_fft);

N_theta = numel(theta);
ind =f_axis>=f_range(1) & f_axis<=f_range(2);
N_proc = numel(f_axis(ind));
f_proc = f_axis(ind)';
I_BF = zeros(N_proc, N_theta);
for n = 1:length(loc)
    M=S(ind,n);
    I_BF = I_BF + sqrt(length(loc))\diag(M) * exp(-2i*pi*f_proc*sind(theta)*loc(n)/c);
end
I_BF = abs(sum(I_BF));
end

