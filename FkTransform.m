function y = FkTransform(n,kelm,x,mode)
% FkTransform
if strcmp(mode,'notransp')      % compute R*x 
   x = reshape(x,n,kelm);       % transform vector into matrix 
   z = fft2(x)/sqrt(n*n);       % apply normalized 2d fast Fourier transform 
   z = fftshift(z);             % re-order the Fourier models
   y = z;
else
   z = zeros(n,kelm);           % back to multichannel time domain
   z = x;
   z = ifftshift(z);
   y = ifft2(z)*sqrt(n*n);
   y = reshape(y,n*kelm,1);
end