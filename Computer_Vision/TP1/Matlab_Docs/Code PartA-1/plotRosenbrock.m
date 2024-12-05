function plotRosenbrock(X)

cla reset;
x=-3:.02:2;
y=-1:.02:3;
[xx,yy]=meshgrid(x,y);
zz=100*(yy-xx.^2).^2+(1-xx).^2;
surface(x,y,zz,'EdgeColor','none'); % hide the edges
view(0,90); % overhead view 
xlabel('x');
ylabel('y');
axis equal;
axis([-3 2 -1 3]);
hsv2=hsv;
hsv3=[hsv2(11:64,:); hsv2(1:10,:)];
colormap(hsv3); % different colormap for 2 axes
hold on;

plot3(X(1),X(2),10000,'bo', ...
    'MarkerSize',5, ...
    'EraseMode','none');
title(['Current X Value: [' num2str(X(1)) ' ' num2str(X(2)) ']']);
drawnow;

end
