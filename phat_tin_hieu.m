% Thêm đường dẫn cần thiết
addpath('D:\GNSS\GNSS_SDR_IQ\include');

% Tạo các mã C/A cho PRNs từ 11 đến 20 và ghép chúng lại thành một mảng 1x10230
cacode11 = generateCAcode(11);
cacode12 = generateCAcode(12);
cacode13 = generateCAcode(13);
cacode14 = generateCAcode(14);
cacode15 = generateCAcode(15);
cacode16 = generateCAcode(16);
cacode17 = generateCAcode(17);
cacode18 = generateCAcode(18);
cacode19 = generateCAcode(19);
cacode20 = generateCAcode(20);
cacodes = [cacode11, cacode12, cacode13, cacode14, cacode15, cacode16, cacode17, cacode18, cacode19, cacode20];

Rc = 1.023e6;
fs = 8e6;
ft=2;
time = 10; % 10 giây
N = fs * time;
n = 0:N-1;
tau1 = 40;
tau2 = 100;
n1 = n + tau1;

r = cacodes(rem(floor(n1 / fs * Rc), 10230) + 1);
normalized_IQ = r * 1024;
normalized_I = normalized_IQ .* cos(2 * pi * ft * n / fs);
normalized_Q = normalized_IQ .* sin(2 * pi * ft * n / fs);

IQ = zeros(1, 2 * length(r));
IQ(1:2:end) = normalized_I;
IQ(2:2:end) = normalized_Q;

num_repeats = 5;
IQ_p_repeated = repmat(IQ, 1, num_repeats);

fid_11_20 = fopen(['D:\do_an_tot_nghiep\data_2506_phat_ca_new2_8M.bin'], 'wb');

if fid_11_20 == -1
    error('Không thể mở tệp tin để ghi.');
end

r_p_int = int16(IQ_p_repeated);
fwrite(fid_11_20, r_p_int, 'int16');
fclose(fid_11_20);
