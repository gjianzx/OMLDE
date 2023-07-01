% % ============================================ % %
% % An enhanced differential evolution algorithm with a new oppositional-mutual learning strategy 
% % ============================================ % %

function [Pb,trace,FEs_fitness]=OMLDE(func_num,fhd,D,NP,F,CR,gen_max,Max_FES,border,varargin)

fbias=[100,200,300,400,500,600,700,...
       800,900,1000,1100,1200,1300,...
       1400,1500,1600,1700,1800,1900,...
       2000,2100,2200,2300,2400,2500,...
       2600,2700,2800,2900,3000];
trace=zeros(gen_max,2);
bounds=border*ones(D,2);
bounds(:,1)=-1*bounds(:,1);
rng=(bounds(:,2)-bounds(:,1))';    
x=(ones(NP,1)*rng).*(rand(NP,D))+(ones(NP,1)*bounds(:,1)');

aa=max(x);
bb=min(x);
for i=1:NP    
    r1=rand;r2=rand;r3=rand;
    if rand<0.5
        ox(i,:)=x(i,:)+r1*(r2*(aa+bb-x(i,:))-x(i,:));
%         ox(i,:)=x(i,:)+rand(1,D).*(rand(1,D).*(aa+bb-x(i,:))-x(i,:));
    else
        j=randi([1,NP],1,1);
        while i==j
            j=randi([1,NP],1,1);
        end
        ox(i,:)=x(i,:)+r3*(x(j,:)-x(i,:));
%          ox(i,:)=x(i,:)+rand(1,D).*(x(j,:)-x(i,:));
    end
    for k=1:D
        if (ox(i,k)>aa(k))    
            ox(i,k)=aa(k);
        end
        if (ox(i,k)<bb(k))
            ox(i,k)=bb(k);
        end
    end
    if (feval(fhd,ox(i,:)',varargin{:})-fbias(func_num))<(feval(fhd,x(i,:)',varargin{:})-fbias(func_num))
        x(i,:)=ox(i,:);
    end
end

trial=zeros(1,D);
cost=zeros(1,NP);

Pb=inf; 
Xb=x(1,:);
for i=1:NP   
    cost(i)=feval(fhd,x(i,:)',varargin{:})-fbias(func_num);
    if(cost(i)<=Pb)
        Pb=cost(i);
        Xb=x(i,:);                
   end
end

fitFEs_count = NP;
initial_FEs = 1;
new_FEs = fitFEs_count;
FEs_fitness(initial_FEs:new_FEs) = Pb;
old_FEs = new_FEs;

trace(1,1)=1;
trace(1,2)=Pb;

for count = 2 : gen_max
    
    if fitFEs_count > Max_FES
        break;
    end
    for i=1:NP
        while 2>1
            a=floor(rand*NP)+1;   %floor(X) 将 X 的每个元素四舍五入到小于或等于该元素的最接近整数。
            if a~=i
                break;
            end
        end
        while 2>1
            b=floor(rand*NP)+1;
            if b~=i&&b~=a
                break;
            end
        end
        while 2>1
            c=floor(rand*NP)+1;
            if c~=i&&c~=a&&c~=b
                break;
            end
        end
        
        jrand=floor(rand*D+1);
        aa=max(x);
        bb=min(x);
        for k=1:D
            if(rand<CR||jrand==k)     
                trial(k)=x(c,k)+F*(x(a,k)-x(b,k)); 
            else
                trial(k)=x(i,k);
            end
            if trial(k)<bb(k)
                 trial(k)=bb(k);
            end
            if trial(k)>aa(k)
                 trial(k)=aa(k);
            end
        end
    
    if rand<=0.05       
            r1=rand;r2=rand;r3=rand;
            if rand<0.5
                ox(i,:)=x(i,:)+r1*(r2*(aa+bb-x(i,:))-x(i,:));
%                 ox(i,:)=x(i,:)+rand(1,D).*(rand(1,D).*(aa+bb-x(i,:))-x(i,:));
            else
                j=randi([1,NP],1,1);
                while i==j
                    j=randi([1,NP],1,1);
                end
                ox(i,:)=x(i,:)+r3*(x(j,:)-x(i,:));
%                 ox(i,:)=x(i,:)+rand(1,D).*(x(j,:)-x(i,:));    
            end
            for k=1:D
                if (ox(i,k)>aa(k))   
                    ox(i,k)=aa(k);
                end
                if (ox(i,k)<bb(k))
                    ox(i,k)=bb(k);
                end
            end
    end
            oxscore=feval(fhd,ox(i,:)',varargin{:})-fbias(func_num);
            trialscore=feval(fhd,trial(:),varargin{:})-fbias(func_num); 
            fitFEs_count = fitFEs_count + 1;    
            xiscore=cost(i);

            xmin=xiscore;
            if xmin>trialscore
                x(i,1:D)=trial(1:D);
                cost(i)=trialscore;
                xmin=trialscore;
            end
            if xmin>oxscore
                x(i,1:D)=ox(1:D);
                cost(i)=oxscore;
            end
            if cost(i)<=Pb
                Pb=cost(i);
                if cost(i)<=eps
                    cost(i)=0;
                end                
                Xb(1:D)=x(i,1:D);
            end 
    new_FEs = fitFEs_count;
    FEs_fitness(old_FEs:new_FEs) = Pb;
    old_FEs = new_FEs;            
   end

    trace(count,1)=count;
    trace(count,2)=Pb;
end

end