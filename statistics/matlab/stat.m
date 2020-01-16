data_pf = table2array(readtable('csv/map10/MonteCarloStatistics_pf_map10v2_kidnepped.csv'));
data_de = table2array(readtable('csv/map10/MonteCarloStatistics_de_map10v2_kidnepped.csv'));
mean_pf = mean(data_pf,1);
mean_pf = mean_pf(2:end)
mean_de = mean(data_de,1);
mean_de = mean_de(2:end)
x = 1:length(mean_pf);
hold on;
plot(x, mean_pf,'b','LineWidth', 1);
plot(x, mean_de,'r','LineWidth', 1);
xlabel('k');
ylabel('MSE');
% ---- map5 ---- %
% xlim([1,96]);
% ylim([0,14.5])

% ---- map3 ---- %
% xlim([1,196]);
% ylim([0,14.5])

% ---- map10 ---- %
xlim([1,109]);
ylim([0,20.5])
