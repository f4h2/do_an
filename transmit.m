clear all;
close all;
addpath('D:\GNSS\GNSS_SDR_IQ\include');
speedOfLight = 299792458;
samplingFreq=10e6;
codeLength =1023;
codeRate=1.023e6;
NumTr=3;%Số lượng trạm phát
numSamp=0.01*samplingFreq;

caCode=generateCAcode(1);
prnCode(1,:)=[caCode caCode];

caCode=generateCAcode(2);
prnCode(2,:)=[caCode caCode];

caCode=generateCAcode(3);
prnCode(3,:)=[caCode caCode];

%Giả sử tọa độ 3 trạm phát
Xt(1,:)=[0,1];
Xt(2,:)=[549,0];
Xt(3,:)=[0,545];

%Giả sử tọa độ bộ thu
Xr=[430,280];

codePhase   = codeRate/samplingFreq;
%Tính toán k/c từ bộ thu đến các trạm phát
for i=1:NumTr
    R(i)=sqrt((Xt(i,1)-Xr(1))^2+(Xt(i,2)-Xr(2))^2);
    Rt(i)=R(i)/speedOfLight;
    Rn(i)=Rt(i)*samplingFreq;
end

signal_1Block=zeros(1,numSamp);
for i=1:NumTr
    tcode = [0:numSamp-1]*codePhase -Rt(i)*codeRate;     
    tcode2=floor(rem(tcode+1023,1023))+1;                
    code=prnCode(i,tcode2);                              
    signal_1Block =signal_1Block+code;                  
end
signal=1024*signal_1Block;                               
signalIQ=zeros(1,2*length(signal));
signalIQ(1:2:end)=signal;

fid= fopen('test.bin','wb');
fwrite(fid,signalIQ,'int16');
fclose(fid);



