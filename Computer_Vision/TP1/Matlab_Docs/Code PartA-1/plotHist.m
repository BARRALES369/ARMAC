function plotHist(histout)

subplot(2,1,1);
semilogy(histout(:,1),histout(:,2));%logatithmic scale
xlabel('Iterations');
ylabel('Objective function');
grid on;
subplot(2,1,2);
semilogy(histout(:,1),histout(:,3));
xlabel('Iterations');
ylabel('Gradient norm');
grid on;

end
