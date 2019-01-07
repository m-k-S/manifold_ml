%%% Plot hyperboloid
v1 = -5:0.2:5;
v2 = -5:0.2:5;

[x1,x2] = meshgrid(v1,v2);

P = [x1(:) x2(:)];
P = P';

x0 = sqrt(1+ diag(P'*P));

X = [x0';P];

f = figure;
plot3(X(2,:),X(3,:),X(1,:),'.');
axis equal;
xlabel('x1');
ylabel('x2');
zlabel('x0');


%%%%% plot points

% x = [-0.93507078,  0.31983479;
%      -0.59733273,  0.51757949;
%      -0.33056023,  0.64212928;
%      -0.08313685,  0.73130841;
%      0.24213987, 0.8160301;
%      0.63517249, 0.88100614;
%      1.24427689, 0.93662058; ];
 
x = [-0.93507078,  0.31983479;
    -0.41768291,  0.32771861;
    0.16997114, 0.44070757;
    0.36278612, 0.50746986;
    1.0226206 , 0.81634693;
    1.02361635, 0.81687279;
    1.24427689, 0.93662058; ];
 
 
 x0 = sqrt(1+ diag(x*x'));
 
figure(f);
hold on;
plot3(x(:,1),x(:,2),zeros(size(x,1),1),'x-r');
h = plot3(x(:,1),x(:,2),x0,'x-r');
set(h,'linewidth',2);
axis equal;