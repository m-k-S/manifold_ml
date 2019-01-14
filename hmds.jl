using PyCall
@pyimport numpy as np
@pyimport networkx as nx
@pyimport scipy.sparse.csgraph as csg

function power_method(A,d,tol;verbose=false, T=1000)
    (n,n) = size(A)
    x_all = qr(randn(n,d))[1]
    _eig  = zeros(d)
    if verbose
        println("\t\t Entering Power Method $(d) $(tol) $(T) $(n)")
    end
    for j=1:d
        if verbose tic() end
        x = view(x_all,:,j)
        x /= norm(x)
        for t=1:T
            x = A*x
            if j > 1
                #x -= sum(x_all[:,1:(j-1)]*diagm(,2)
                yy = vec(x'view(x_all, :,1:(j-1)))
                for k=1:(j-1)
                    x -= view(x_all,:,k)*yy[k]
                end
            end
            nx = norm(x)
            x /= nx
            cur_dist = abs(nx - _eig[j])
            if !isinf(cur_dist) &&  min(cur_dist, cur_dist/nx) < tol
                if verbose
                    println("\t Done with eigenvalue $(j) at iteration $(t) at abs_tol=$(Float64(abs(nx - _eig[j]))) rel_tol=$(Float64(abs(nx - _eig[j])/nx))")
                end
                if verbose toc() end
                break
            end
            if t % 500 == 0 && verbose
                println("\t $(t) $(cur_dist)\n\t\t $(cur_dist/nx)")
            end
            _eig[j]    = nx
        end
        x_all[:,j] = x
    end
    return (_eig, x_all)
end

function power_method_sign(A,r,tol;verbose=false, T=1000)
    _d, _U    = power_method(A'A,r, tol;T=T)
    X         = _U'A*_U
    _d_signed = vec(diag(X))
    if verbose
        print("Log Off Diagonals: $( Float64(log(vecnorm( X - diagm(_d_signed)))))")
    end
    return _d_signed, _U
end

# hMDS with one PCA call
function h_mds(Z, k, n, tol)
    # run PCA on -Z
    # tic()
    lambdasM, usM = power_method_sign(-Z,k,tol)
    lambdasM_pos = copy(lambdasM)
    usM_pos = copy(usM)

    idx = 0
    for i in 1:k
        if lambdasM[i] > 0
            idx += 1
            lambdasM_pos[idx] = lambdasM[i]
            usM_pos[:,idx] = usM[:,i]
        end
    end

    Xrec = usM_pos[:,1:idx] * diagm(lambdasM_pos[1:idx].^ 0.5);
    # toc()

    return Xrec', idx
end
