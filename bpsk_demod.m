clear all;
addpath('D:\BKIT2NguyenPhong\IT4999\Sample\include');
clc;
%Code
Rc=1.023e6;
Nc=1023;
fs = 10e6; 
ft = 10;
cacodes1=generateCAcode(1);

%Đọc 8 bit
N=8*0.02*fs;% 8 bits, 0.02s mỗi bit
n=(0:N-1);

fid= fopen('signal.bin','rb');


for(i=1:8*20)
    %Lặp lại từng ms
    N1ms=0.001*fs;
    n=[0:N1ms-1];
    lcn1=cacodes1(rem(floor(n/fs*Rc),Nc)+1);
    
    signal= fread(fid,N1ms,'int16');
    signal=signal';
    data_demod(i)=sum(signal.*lcn1);
end
plot(data_demod);
fclose(fid);

