data_pf = table2array(readtable('csv/map10/MonteCarloStatistics_pf_map10v3.csv'));
data_de = table2array(readtable('csv/map10/MonteCarloStatistics_de_map10v3.csv'));
data_pf = data_pf(:,2:end);
data_de = data_de(:,2:end);
mean_pf = mean(data_pf,1);
mean_de = mean(data_de,1);
s_pf = sum(data_pf,2);
best_pf = data_pf(find(s_pf==min(s_pf)),:);
s_de = sum(data_de,2);
best_de = data_de(find(s_de==min(s_de)),:);

x = 1:length(mean_pf);
hold on;
plot(x, mean_pf,'b','LineWidth', 1);
plot(x, mean_de,'r','LineWidth', 1);
% plot(x, best_pf,'b--','LineWidth', 1);
% plot(x, best_de,'r--','LineWidth', 1);
xlabel('k');
ylabel('MSE');
% ---- map5 ---- %
% xlim([1,96]);
% ylim([0,14.5]);

% ---- map3 ---- %
% xlim([1,196]);
% ylim([0,14.5]);

% ---- map4 ---- %
% xlim([1,60]);
% ylim([0,16]);

% ---- map10 ---- %
xlim([1,70]);
ylim([0,32]);
