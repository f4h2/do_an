addpath('D:\BKIT2NguyenPhong\IT4999\Sample\include');
clc;
%Code
Rc=1.023e6;
Nc=1023;
fs = 10e6; 
ft = 10;
speedOfLight = 299792458;
cacodes1=generateCAcode(1);

%Data
Rd=50;% 50 bits/s
dataBits=[0 1 0 1 1 0 0 1];
dataBits=2*dataBits-1; % 0-> -1, 1 -> 1

N=8*0.02*fs;% 8 bits, 0.02s mỗi bit
n=(0:N-1);

lcn1=cacodes1(rem(floor(n/fs*Rc),Nc)+1);
data=dataBits(floor(n/fs*Rd)+1);
signal=lcn1.*data;

fid= fopen('signal.bin','wb');
fwrite(fid,signal,'int16');
fclose(fid);
