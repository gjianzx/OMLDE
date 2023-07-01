clear;
clc;
close all; 
runNumber=30;          % The number of independent runs 
D=10;                  % The problem dimension
NP=100;                % Population size
F=0.5;                 % Mutation factor
CR=0.9;                % Crossover factor 
Max_FES = 10000 * D;   % Maximum number of function evaluations
gen_max = Max_FES/NP;  % Maximum number of algorithm iterations 
border=100;            % Search space boundaries
fhd=str2func('cec17_func'); % CEC2017 Benchmark Functions
str = "OMLDE";         % Algorithm name

global fbias
% Optimal Value Offset %
fbias=[100,200,300,400,500,600,700,...
       800,900,1000,1100,1200,1300,...
       1400,1500,1600,1700,1800,1900,...
       2000,2100,2200,2300,2400,2500,...
       2600,2700,2800,2900,3000];
   
OMLDEMatrix=zeros(runNumber,Max_FES);
Algs_FES = zeros(1,Max_FES);
s=zeros(1,runNumber);
for k=1:30         
    func_num=k;
    if k==2
       continue;
    end
    fprintf("\n----------------------------------\n");
    fprintf("Start testing the %d dimensional-F%d function of %s >>>>\n",D,k,str);
    fprintf("----------------------------------\n");
for i=1:runNumber
    [Pb,~,FEs_fitness]=OMLDE(func_num,fhd,D,NP,F,CR,gen_max,Max_FES,border,func_num);
    OMLDEMatrix(i,:)=FEs_fitness;
    s(1,i)=Pb;
end
Algs_FES(1,:)= mean(OMLDEMatrix,1);
fprintf("\nOMLDE:\nBest为:%d\nWorst为:%d\nMedian为:%d\nMean为:%d\nStd为:%d\n",min(s(1,:)),max(s(1,:)),median(s(1,:)),mean(s(1,:)),std(s(1,:)));
 
end