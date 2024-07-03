addpath('D:\GNSS\GNSS_SDR_IQ\include');
clc;

Rc=1.023e6;
fs = 10e6; 
ft = 10;
speedOfLight = 299792458;
fid = fopen("data_thu_2606_thuc_nghiem_21.bin", 'rb'); 
%fseek(fid,fs*5*2,'bof');
if fid == -1
    error('Không thể mở tệp tin để đọc.');
end

num_samples = round(10e-3 * fs);
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
cacodes1 = [cacode11, cacode12, cacode13, cacode14, cacode15, cacode16, cacode17, cacode18, cacode19, cacode20];

cacode21 = generateCAcode(21);
cacode22 = generateCAcode(22);
cacode23 = generateCAcode(23);
cacode24 = generateCAcode(24);
cacode25 = generateCAcode(25);
cacode26 = generateCAcode(26);
cacode27 = generateCAcode(27);
cacode28 = generateCAcode(28);
cacode29 = generateCAcode(29);
cacode30 = generateCAcode(30);
cacodes2 = [cacode21, cacode22, cacode23, cacode24, cacode25, cacode26, cacode27, cacode28, cacode29, cacode30];

tauMax=fs*0.01;
Nmax=10000;
n=0:Nmax-1;

TX1=[20.9896100,105.7110745];
TX2=[20.9924397,105.7106347];
RX=[20.9911114,105.7107914];

T1=calcDistance(TX1(1),TX1(2),RX(1),RX(2));
T2=calcDistance(TX2(1),TX2(2),RX(1),RX(2));
T0=(T1-T2)/speedOfLight*fs;
D= calcDistance(TX1(1),TX1(2),TX2(1),TX2(2));
disp(["T1:",num2str(T1)]);
disp(["T2:",num2str(T2)]);
disp(["D:",num2str(D)]);

tmp = fread(fid, 2*num_samples, 'int16');
tmp=tmp(1:2*Nmax)';
IQ=tmp(1:2:end)+1i*tmp(2:2:end);
n=(1:Nmax);
IQ=IQ.*exp(1i*2*pi*ft*n/fs);
for(tau=0:tauMax)

    lcn1=cacodes1(rem(floor((n+tau)/fs*Rc),10230)+1);
    Xcorr1(tau+1)=sum(lcn1.*IQ(1:Nmax));

    lcn2=cacodes2(rem(floor((n+tau)/fs*Rc),10230)+1);
    Xcorr2(tau+1)=sum(lcn2.*IQ(1:Nmax));
end
    
[vale1 taue1]=max(Xcorr1);
[vale2 taue2]=max(Xcorr2);

Delta_T=(taue1-taue2)-T0;
Delta_T=Delta_T/fs*speedOfLight;
for(ii=1:1)
    tmp = fread(fid, 2*num_samples, 'int16');
    tmp=tmp(1:2*Nmax)';
    IQ=tmp(1:2:end)+1i*tmp(2:2:end);
    n=(1:Nmax);
    IQ=IQ.*exp(1i*2*pi*ft*n/fs);
    for(tau=0:tauMax)
    
        lcn1=cacodes1(rem(floor((n+tau)/fs*Rc),10230)+1);
        Xcorr1(tau+1)=sum(lcn1.*IQ(1:Nmax));
    
        lcn2=cacodes2(rem(floor((n+tau)/fs*Rc),10230)+1);
        Xcorr2(tau+1)=sum(lcn2.*IQ(1:Nmax));
    end
    
    [vale1 taue1]=max(Xcorr1);
    [vale2 taue2]=max(Xcorr2);

    C(ii)=(taue1-taue2);
    Delta_C(ii)=C(ii)/fs*speedOfLight;
    X(ii)=(-Delta_C(ii)+D+Delta_T)/2;
    X2 = D-X(ii);
    disp(["T1 được tính ra bằng:", num2str(X(ii))]);
    disp(["T2 được tính ra bằng:", num2str(X2)]);
end
fclose(fid);
