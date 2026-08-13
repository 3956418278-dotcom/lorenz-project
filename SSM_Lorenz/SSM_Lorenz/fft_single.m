% Compute the single-sided frequency spectrum of a time domain signal
function X = fft_single(x)
Fs = 16;
tg = 4096;
N1 = Fs*tg;

X = fft(x,[],2) / N1;
X = [X(:,1), 2*X(:,2:N1/2), X(:,N1/2+1)];
end