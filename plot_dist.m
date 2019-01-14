function fig = plot_dist(X,L)
    [~,d] = size(X);
    
    if d>3; warning('data has more than tree dimensions, will only visualize the first three coords'); end
    if d<2; error('data too trivial to visualize'); end
    
    style = {'bx','ro','g.','cs','m*','kd','y.'};
    
    ul = unique(L);
    
    fig = figure;
    hold on;
    for i=1:length(ul)
        x = X(L==ul(i),:);
        switch d
            case 2
                plot(x(:,1),x(:,2),style{i});
            otherwise
                plot3(x(:,1),x(:,2),x(:,3),style{i});
        end
        
    end
    axis equal;
end