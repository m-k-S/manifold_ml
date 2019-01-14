
% hMDS with one PCA call
function [Xrec, sidx] = hmds(Z, k, n, tol)

tmp1 = cellfun(@cell2mat,Z,'uniformoutput',false);
tmp2 = [tmp1{:}]; %<<<

Z = reshape(tmp2, [sqrt(length(tmp2)) sqrt(length(tmp2))]);
Z = double(Z);

    % run PCA on -Z
    % tic()
    [lambdasM, usM] = power_method_sign(-Z,k,tol);

%    lambdasM_pos = lambdasM;
%    usM_pos = usM;
%
%    idx = 0;
%    for i = 1:k
%        if lambdasM(i) > 0
%            idx = idx+1;
%            lambdasM_pos(idx) = lambdasM(i);  %<<<
%            usM_pos(:,idx) = usM(:,i);        %<<<<
%        end
%    end

    [~,sidx] = sort(lambdasM,'descend');

    Xrec = usM(:,sidx(1:k)) * diag(lambdasM(sidx(1:k)).^ 0.5);

%    Xrec = usM_pos(:,1:idx) * diag(lambdasM_pos(1:idx).^ 0.5);
    % toc()

    %Xrec_transpose = Xrec';
    %return Xrec', idx;
end


function [d_signed, U_] = power_method_sign(A,r,tol,verbose,T)
    if nargin<4; verbose = false; end;
    if nargin<5; T=1000; end;
%A = real(A);
%class(real(A))


    [d_, U_] = power_method(A'*A, r, tol, T);
    X        = U_'*A*U_;  % matrix
    %d_signed = vec(diag(X))
    d_signed = diag(X);

  %if verbose
  %      fprintf('Log Off Diagonals: $( Float64(log(vecnorm( X - diagm(d_signed)))))')
  %  end
  %  return d_signed, U_
end

function [eig_, x_all] = power_method(A, d, tol, verbose, T)
    if nargin<4; verbose = false; end;
    if nargin<5; T=1000; end;

      %(n,n) = size(A)
    [n1,n2] = size(A);
    assert(n1==n2);
    n=n1;

    %print("break1")
    [x_all,~] = qr(randn(n,d));  %[1]

    eig_  = zeros(d,1); %   eig_  = zeros(d);
    %if verbose
    %    println("\t\t Entering Power Method $(d) $(tol) $(T) $(n)")
    %end
    for j=1:d
        %if verbose tic() end
        x = x_all(:,j);  %x = view(x_all,:,j)
        x = x/norm(x);

        for t=1:T
            x = A*x;
            if j > 1
                %%%%%%   x -= sum(x_all[:,1:(j-1)]*diagm(,2)
                yy = x' * x_all(:, 1:j-1); %   vec(x' view(x_all, :,1:(j-1)))
                for k=1:(j-1)
                    x = x - x_all(:,k)*yy(k);  % x -= view(x_all,:,k)*yy[k]
                end
            end
            nx = norm(x);
            x = x/nx;
            cur_dist = abs(nx - eig_(j));
            %if !isinf(cur_dist) &&  min(cur_dist, cur_dist/nx) < tol
            if ~isinf(cur_dist) &&  min(cur_dist, cur_dist/nx) < tol
                %if verbose
                %    println("\t Done with eigenvalue $(j) at iteration $(t) at abs_tol=$(Float64(abs(nx - _eig[j]))) rel_tol=$(Float64(abs(nx - _eig[j])/nx))")
                %end
                %if verbose toc() end
                break;
            end
            %if mod(t,500) == 0 && verbose
            %    println("\t $(t) $(cur_dist)\n\t\t $(cur_dist/nx)")
            %end
            eig_(j)    = nx;
        end
        x_all(:,j) = x;
    end
    %return (_eig, x_all)
end
