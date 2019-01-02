function [E,A,T,L,G] = gen_rand_tree(num_nodes, style, num_labels, label_style)
%
%  usage    [E,A,T] = gen_rand_tree(num_nodes, style, num_labels, label_style);
%
%  style = 'bal'   (balanced degree)
%        = 'pref'  (preferred to be attached to higher degree)
%        = 'unif'  (attached to previous nodes uniformly)
%
%  label_style = 'unif' (all nodes are labelled uniformly)
%                'hier' (labelled randomly respecting the hierarcy)

if nargin<1
    num_nodes = 1;
end
if nargin<2
    style = 'bal';
end
if nargin<3
    num_labels = 1;
end
if nargin<4
    label_style = 'hier';
end


A = zeros(num_nodes);
T = zeros(1,num_nodes);
L = zeros(num_nodes,1);

for n=2:num_nodes
    if n==2
        A(1,2) = 1; 
        p = 1;
    else
        d = sum(A,1);
        d=d(1:n-1);
        
        switch style
            case 'unif'
                prb = ones(size(d))/length(d);
            case 'bal'
                prb = (1./d)/sum(1./d);
            case 'pref'
                prb = d/sum(d);
            otherwise
                error('incorrect style');
        end
        p = sum(cumsum(prb)<rand)+1;

        A(n,p)=1;
    end
    T(n) = p;
    A = A+A';
    A(A~=0) = 1;
end

[E1,E2] = ind2sub(size(A),find(A~=0));
E = [E1 E2];

G = graph(A);

%%% labels

switch label_style
    case 'hier'
        % seed
        prm = randperm(num_nodes,num_labels);
        L(prm) = (1:num_labels)';
        
        
        for n=1:num_nodes
            if L(n)~=0; continue; end
            
            d = zeros(1,length(prm));
            for i=1:length(d)
                d(i) = length(shortestpath(G,n,prm(i)));
            end
            d = 1./(d.^8); prb = d/sum(d);
            p = sum(cumsum(prb)<rand)+1;
            %[~,p] = min(d);
            L(n) = p;
        end
        
    case 'unif'
        L = randi(num_labels,num_nodes,1);
    otherwise
        error('incorrect label style');
end

