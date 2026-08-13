% Generation of a sinusoidal foring given its amplitude, phase and
% frequency
function f = sine_force(Amp,fai,fre)
    ts = 30;
    tg = 4096;
    Fs = 16;
    N1 = Fs*tg;
    
    m = size(Amp,1);
    f = zeros(m, Fs*(ts+tg));

    if nargin < 3 || isempty(fai)
        seed = 0;
        rng(seed,"twister")
        fai = rand(m,N1/2)*2*pi;
        fai = [zeros(m,1), fai];
    end

    % Series of ts ≤ t ≤ ts+tg: use inverse fft
    F = Amp.*exp(1i*(fai + 2*pi/tg * repmat(0:N1/2,m,1)*ts)); % Frequency spectrum of the input
    F(:,setdiff(1:(N1/2+1),fre+1)) = 0; % Force on estimated frequencies
    F(:,end) = real(F(:,end)); % Deal with the critical cases
    F = [F(:,1), F(:,2:end-1)/2, F(:,end), fliplr(conj(F(:,2:end-1)))/2]*N1; % Convert to two side spectrum
    f(:, Fs*ts+1:end) = ifft(F,[],2); % Use inverse fft to generate time domain input
    
    % Series of 0 ≤ t ≤ ts: compute directly
    for l = 1:m
        for j = 1:(Fs*ts)
            f(l,j) = Amp(l,fre+1) * (cos(2*pi/tg * fre * (j-1)/Fs + fai(l,fre+1)))';
        end
    end
end