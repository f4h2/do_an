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

fid= fopen('test.bin','rb');
signalIQ= fread(fid,2*numSamp,'int16');
fclose(fid);
signal=signalIQ(1:2:end);
signal=signal';

%Receiver Processing 
figure(100); hold on;
numProcessedSamp=50000;
codePhase   = codeRate/samplingFreq;                       
for i=1:NumTr
    for tau=0:codeLength/codeRate*samplingFreq                    
        tcode = ([0:numProcessedSamp-1]+ tau)*codePhase;          
        tcode2=floor(rem(tcode,1023))+1;
        code=prnCode(i,tcode2);
        Xcorr(i,tau+1)=sum(code.*signal(1:numProcessedSamp))^2;     
    end
    [val,idx]=max(Xcorr(i,:));
    Tn(i)=idx;
    disp(sprintf('%d of sv %d',idx,i));
    plot(Xcorr(i,:));
end
for(i=1:NumTr)
    T(i)=max(Tn)-Tn(i);                                              
    P(i)=(5+T(i))/samplingFreq*speedOfLight;                        

end


pos     = zeros(3, 1);                                   
obs=P;                          
numOfIterations=10;
F= zeros(NumTr, 1);
J= zeros(NumTr, 3);
for iter = 1:numOfIterations
    for i = 1:NumTr
        F(i)=obs(i)-sqrt((Xt(i,1)-pos(1))^2+(Xt(i,2)-pos(2))^2)-pos(3);
        J(i, :) =  [ (-(Xt(i,1) - pos(1))) / obs(i) ...
                     (-(Xt(i,2) - pos(2))) / obs(i) ...
                     1 ];
    end 
    delta_pos   = J \ F;%J^(-1)*F
    pos = pos + delta_pos;                           
end
disp(['Vị trí bộ thu: ', num2str(pos(1:2, :)')]);