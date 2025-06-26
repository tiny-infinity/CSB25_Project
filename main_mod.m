clear

%%
signal=[0 0 0];  %% Signal

parameter_sets = readmatrix("MISA_parameters.csv");
bi_stable_para_sets = parameter_sets(parameter_sets(:,2) == 2,3:end);

action_mat = zeros(size(bi_stable_para_sets,1),2);

for para_indx = 4%:size(bi_stable_para_sets,1) % set para_indx to 4 and 25 to reproduce MATLAB figure mentioned in the email 
par = bi_stable_para_sets(para_indx,:); %% The parameters of the ODE

soln_set = readmatrix("MISA_solution.csv");
bi_stable_soln_set = soln_set(soln_set(:,2) == 2,:);
soln = 2.^(bi_stable_soln_set((para_indx-1)*2+1:para_indx*2,4:end));

%% refinning the racipe solutions
soln_upd = zeros(2,size(soln,2));
for soln_indx = 1:2
    x0 = soln(soln_indx,:);
    x1 = inf*ones(1,size(soln,2));
    tot_time = 0;
    err = 1;
    while (err>1e-7 && tot_time < 2000)
        [t,y]=ode45(@(t,x)MISA(t,x,par,signal),[0,100],x0);
        x1 = y(end,:);
        err = norm(x1-x0,2);
        x0 = x1;
        tot_time = tot_time + 100;
    end
    soln_upd(soln_indx,:) = x0;
end
%% Solve the ODEs, calculate the paths and actions;
[ycell,action]=Solver(par,signal,soln_upd');

action_mat(para_indx,:) = [action(1,2) action(2,1)];

end

% plotting transition path
figure('Position',[680 463 991 415]);
subplot(1,2,1)
plot(ycell{1,2}(1,:),ycell{1,2}(2,:))
hold on
plot(ycell{1,2}(1,1),ycell{1,2}(2,1),'Marker','o','MarkerSize',10,'MarkerFaceColor','c');
plot(ycell{1,2}(1,end),ycell{1,2}(2,end),'Marker','sq','MarkerSize',10,'MarkerFaceColor','w');
title(['Action ' num2str(round(action_mat(para_indx,1)),2)])
legend('Path B to A', 'start','end');
xlabel('A exp')
ylabel('B exp')
ax = gca;
ax.FontSize = 14;
grid on

subplot(1,2,2)
plot(ycell{2,1}(1,:),ycell{2,1}(2,:))
hold on
plot(ycell{2,1}(1,1),ycell{2,1}(2,1),'Marker','o','MarkerSize',10,'MarkerFaceColor','c');
plot(ycell{2,1}(1,end),ycell{2,1}(2,end),'Marker','square','MarkerSize',10,'MarkerFaceColor','w');
title(['Action ' num2str(round(action_mat(para_indx,2)),2)])
legend('Path A to B', 'start','end')
xlabel('A exp')
ylabel('B exp')
ax = gca;
ax.FontSize = 14;
grid on