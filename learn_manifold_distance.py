import numpy as np
from scipy.integrate import quad

# Arc length integrand for the sinusoid manifold: (x, y, sin(x) + sin(y))
def integrand_sinusoid(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x

    r = Pth[0]
    s = Pth[1]
    D = np.asarray([[1, 0], [0, 1], [np.cos(r), np.cos(s)]])

    v = np.matmul(D, np.matmul(Q, Dff))
    return np.linalg.norm(v)

# Arc length integrand for the hyperboloid
def integrand_hyp(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x
    ip = np.matmul(Dff.T, np.matmul(Q.T, np.matmul(Q, Dff)))
    nm = np.matmul(Pth.T, Dff)
    dn = 1 + np.matmul(Pth.T, Pth)

    return np.sqrt( ip - (nm**2 / dn) )

# Arc length integrand for the helicoid
def integrand_helicoid(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x   # 2 x 1

    r = Pth[0]
    s = Pth[1]
    D = [[np.cos(s), -r * np.sin(s)],
        [np.sin(s), r * np.cos(s)],
        [0, 1]]

    v = np.matmul(D, np.matmul(Q, Dff))
    return np.linalg.norm(v)

# Takes points x, y in the base space and computes the distance between their mappings on the given manifold
# Q is a linear transformation on x, y in the base Euclidean space; set to identity if not desired
# The number of segments increases the resolution of the approximation path, but also heavily increases computation time
# Samples increases the number of sampled points at each segment recalculation
def learn_distance(x, y, Q, integrand, samples=100, segments=7):
    dim = len(x)
    path_segments = []
    for i in range(segments):
        path_segments.append((i / (segments-1) * (y - x)) + x)

    convergence = True
    while convergence is True:
        convergence = False

        for idx, segment in enumerate(path_segments[1:-1]):
            i = idx + 1
            this_point = segment
            prev_point = path_segments[i-1]
            next_point = path_segments[i+1]

            sample_radius = max(np.linalg.norm(this_point - prev_point), np.linalg.norm(this_point - next_point))

            s = np.random.uniform(-sample_radius,sample_radius,size=(samples, dim))
            D1 = quad(integrand, 0, 1, args=(prev_point, this_point, Q))[0]
            D2 = quad(integrand, 0, 1, args=(this_point, next_point, Q))[0]

            min_dist = D1+D2
            best_sample = this_point
            for sample in s:
                sample = sample + this_point


                D1 = quad(integrand, 0, 1, args=(prev_point, sample, Q))[0]
                D2 = quad(integrand, 0, 1, args=(sample, next_point, Q))[0]

                Distance = D1 + D2
                if Distance < min_dist:
                    min_dist = Distance
                    best_sample = sample

            if np.linalg.norm(this_point - best_sample) < 10e-6:
                path_segments[i] = best_sample
            else:
                convergence = True
                path_segments[i] = best_sample

    # while loop ends

    total_distance = 0
    for idx in range(len(path_segments[:-1])):
        Distance = quad(integrand, 0, 1, args=(path_segments[idx], path_segments[idx+1], Q))[0]
        total_distance += Distance

    return total_distance
