function plot_all(X,y,X_star,f_star,mean_star,var_star,Z,m)

set(0,'defaulttextinterpreter','latex')

c1 = [43,140,190]/255;
c2 = [166,189,219]/255;
c3 = [99,99,99]/255;
c4 = [189,189,189]/255;
color2 = [217,95,2]/255;


N = size(X,1);
M = size(Z,1);

fig = figure(1);
set(fig,'units','normalized','outerposition',[0 0 1 .5])

clear h;
clear leg;
hold

h(1) = scatter(X,y,20,'filled', 'MarkerFaceAlpha',3/8,'MarkerFaceColor',c2);
[l,h(4)] = boundedline(X_star, mean_star, 2.0*sqrt(var_star), ':','alpha','cmap','transparency',0.2, color2);
outlinebounds(l,h(4));
h(2) = plot(X_star, f_star,'LineWidth',5, 'Color', c1);
h(5) = plot(Z, m,'ro','MarkerSize',20, 'LineWidth',3);
%h(6) = plot(Z, 0.25 + max(y)*ones(size(Z,1), 1), 'k+', 'MarkerSize', 12,'LineWidth',2);
h(3) = plot(X_star,mean_star,'--', 'Color',c3,'LineWidth',3);

leg{2} = '$f(x)$';
leg{1} = sprintf('%d training data', N);
leg{3} = '$\overline{f}(x)$'; leg{4} = 'Two standard deviations';
leg{5} = sprintf('%d hypothetical data', M);

hl = legend(h,leg,'Location','southwest');
legend boxoff
set(hl,'Interpreter','latex')
xlabel('$x$')
ylabel('$f(x), \overline{f}(x)$')


axis tight
ylim(ylim + [-diff(ylim)/10 0]);
xlim(xlim + [-diff(xlim)/10 0]);
set(gca, 'FontSize', 25);
set(gcf, 'Color', 'w');

end