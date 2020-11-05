clc;
clear all;
close all;
load turangcars
%xcal=xlsread('SSS.xlsx',1);
%ycal=xlsread('SSS.xlsx',2);
%xpre=[];
%ypre=[];

ycal = ycal;
Xcal=xcal;
Xval=xpre;
yval=ypre; 
m_min=5;
m_max=61;

N = size(Xcal,1); 
K = size(Xcal,2); 


% Phase 1: ����ѡ���ѡ�Ӽ���ͶӰ����
    
normalization_factor = std(Xcal);
    
for k = 1:K
    x = Xcal(:,k);
    Xcaln(:,k) = (x - mean(x)) / normalization_factor(k);
end


SEL = zeros(m_max,K);
for k = 1:K
    SEL(:,k) = projection(Xcaln,k,m_max);
end

    
% Phase 2: ����PRESS��׼������ѡ�Ӽ�  
% PRESS(prediction errors sum of squares)Ԥ�����ƽ����

PRESS = Inf*ones(m_max,K);

for k = 1:K
    for m = m_min:m_max
        var_sel = SEL(1:m,k);
        [yhat1,e1] = validation(Xcal,ycal,Xval,yval,var_sel);
        PRESS(m,k) = e1'*e1;
    end
end


[PRESSmin,m_sel] = min(PRESS);  %�ҳ�ÿ����Сֵ�����������У���������
[dummny,k_sel] = min(PRESSmin); %�ҵ���������Сֵ

%��k_sel����Ϊ��ʼ����ʱ��ѣ�������ĿΪm_sel(k_sel)
var_sel_phase2 = SEL(1:m_sel(k_sel),k_sel); 


% Phase 3: ������������

% Step 3.1: ���ָ���ļ���
Xcal2 = [ones(N,1) Xcal(:,var_sel_phase2)]; 
b = Xcal2\ycal; % MLR with intercept term
std_deviation = std(Xcal2);
relev = abs(b.*std_deviation');
relev1 = relev(2:end);
% ��������ԡ��Ľ������ѡ������������
[dummy,index_increasing_relev] = sort(relev1);
index_decreasing_relev = index_increasing_relev(end:-1:1);
% Step 3.2: ����PRESSֵ
for i = 1:length(var_sel_phase2)
    [yhat2,e2] = validation(Xcal,ycal,Xval,yval,var_sel_phase2(index_decreasing_relev(1:i)) );
    PRESS_scree(i) = e2'*e2;
end
RMSEP_scree = sqrt(PRESS_scree/length(e2));
figure, grid, hold on
plot(RMSEP_scree)
xlabel('Number of variables included in the model'),ylabel('RMSE')

% Step 3.3: F�����׼
PRESS_scree_min = min(PRESS_scree);
alpha = 0.25;
dof = length(e2); % Number of degrees of freedom
fcrit = finv(1-alpha,dof,dof); %�ٽ�Fֵ
PRESS_crit = PRESS_scree_min*fcrit;
% ����PRESS_scree����С������
%������PRESS_scree_min��ö� 
zx=find(PRESS_scree < PRESS_crit);
i_crit = min(zx); 
i_crit = max(m_min,i_crit); 

var_sel1 = var_sel_phase2( index_decreasing_relev(1:i_crit) );
title(['Final number of selected variables: ' num2str(length(var_sel)) '  (RMSE = ' num2str(RMSEP_scree(i_crit)) ')'])

% ָʾ��ʯͼ�ϵ�ѡ����
plot(i_crit,RMSEP_scree(i_crit),'s')


%  ��ʾ��ѡ����
% ��У׼���ĵ�һ��������
figure,plot(Xcal(6,:));hold,grid
plot(var_sel1,Xcal(6,var_sel1),'s')
legend('First calibration object','Selected variables')
xlabel('Variable index')
save boduan_shang var_sel1
%��ָ֤��
% [PRESS,RMSEP,SDV,BIAS,r] = validation_metrics(Xcal,ycal,Xval,yval,var_sel);  