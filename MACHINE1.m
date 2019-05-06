clear all
clear
clc
ds = tabularTextDatastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',18000);
T = read(ds);
TrainingSet=T(1:1080,:);
size(T);
Alpha=.01;

m=length(T{:,1});

U0train=T{1:1080,2};
U0train=(U0train-mean(U0train))./std(U0train);

U0CV=T{1081:1500,2};
U0CV=(U0CV-mean(U0CV))./std(U0CV); %Normalizing output


UTrain=T{1:1080,6:10};
UCV=T{1081:1800,6:10};

LenTrain = length(UTrain);
LenCV = length(UCV);
z=length(UTrain);
XTrain=[ones(z,1) UTrain UTrain.^2];
XCV=[ones(length(UCV),1) UCV UCV.^2];

nTrain=length(XTrain(1,:));
for w=2:nTrain
    if max(abs(XTrain(:,w)))~=0
    XTrain(:,w)=(XTrain(:,w)-mean((XTrain(:,w))))./std(XTrain(:,w));
    end
end
nCV=length(XCV(1,:));
for w=2:nCV
    if max(abs(XCV(:,w)))~=0
    XCV(:,w)=(XCV(:,w)-mean((XCV(:,w))))./std(XCV(:,w));
    end
end

YTrain=TrainingSet{:,3}/mean(TrainingSet{:,3});
YCV=TrainingSet{11,3}/mean(TrainingSet{11,3});

ThetaTrain=zeros(nTrain,1);
ThetaCV=zeros(nCV,1);
% for k=1:500
% 
% ETrain(k)=(1/(2*m))*sum((XTrain*ThetaTrain-Y).^2);
% ECV(k)=(1/(2*m))*sum((XCV*ThetaCV-Y).^2);
% end
k=1;
R=1;
ETrain=[];
ECV=[];
while R==1
Alpha=Alpha*1;
ThetaTrain=ThetaTrain-(Alpha/m)*XTrain'*(XTrain*ThetaTrain-YTrain);
ThetaCV=ThetaCV-(Alpha/m)*XCV'*(XCV*ThetaCV-YCV);

ETrain=(1/(2*m))*sum((XTrain*ThetaTrain-YTrain).^2);
ETrain=[ETrain;ETrain];
ECV(k)=(1/(2*m))*sum((XCV*ThetaCV-YCV).^2);
ECV=[ECV;ECV];
 k=k+1
if ETrain(k-1)-ETrain(k)<0
   
    break
end 
q=(ETrain(k-1)-ETrain(k))./ETrain(k-1);
if q <.000001
    R=0;
end
end

figure (1)
plot(k,ECV,'black')
hold on
plot(k,ETrain,'red')
legend('CV','Train')
title('House Price')
ylabel('Cost Fun')
xlabel('Iter')

