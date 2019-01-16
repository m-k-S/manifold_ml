import numpy as np
from scipy.integrate import quad

def Negatives(x, n):
    samples = []
    for edge in D:
        w = x == edge[0]
        v = x == edge[1]
        if w.all() or v.all():
            samples.append(edge)
        if len(samples) == n:
            break

    return samples

NEGATIVES = 5

def integrand_hyp(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x
    ip = np.matmul(Dff.T, np.matmul(Q.T, np.matmul(Q, Dff)))
    nm = np.matmul(Pth.T, Dff)
    dn = 1 + np.matmul(Pth.T, Pth)

    return np.sqrt( ip - (nm**2 / dn) )

def integrand_helicoid(t, x, y, Q):
    Pth = (1 - t) * x + (y * t)
    Dff = y-x   # 2 x 1

    r = Pth[0]
    s = Pth[1]
    D = [[np.cos(s), -r * np.sin(s)],
        [np.sin(s), r * np.cos(s)],
        [0, 1]]

#[[1 0 0]
#[0 1 0]
#[0 0 1]]
#[0.26626796 0.96126932 0.54656653]

    v = np.matmul(D, np.matmul(Q, Dff))
    #ValueError: shapes (3,2) and (3,) not aligned: 2 (dim 1) != 3 (dim 0)
    return np.linalg.norm(v)
#
# Pth = [r ; s]    ->    r cos(s) r sin(s)  s
#  Q()
#d / dr = cos(s) sin(s) 0
#d/ds   = -r sin(s)   r cos(s) 1
#  D = [d/dr  d/ds]



def learn_distance(x, y, Q, integrand, samples=100, segments=7):
    # x = x[1:]
    # y = y[1:]
    x = np.asarray(x)
    y = np.asarray(y)
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

            # this_point_x0 = np.sqrt(1 + np.dot(this_point.T, this_point))
            # next_point_x0 = np.sqrt(1 + np.dot(next_point.T, next_point))
            # prev_point_x0 = np.sqrt(1 + np.dot(prev_point.T, prev_point))
            # this_point_hyp = [this_point_x0, this_point[0], this_point[1]]
            # next_point_hyp = [next_point_x0, next_point[0], next_point[1]]
            # prev_point_hyp = [prev_point_x0, prev_point[0], prev_point[1]]
            # D1 = dhyp(prev_point_hyp, this_point_hyp)
            # D2 = dhyp(this_point_hyp, next_point_hyp)

            min_dist = D1+D2
            best_sample = this_point
            for sample in s:
                # Add sample to current point
                sample = sample + this_point


                D1 = quad(integrand, 0, 1, args=(prev_point, sample, Q))[0]
                D2 = quad(integrand, 0, 1, args=(sample, next_point, Q))[0]

                # sample_x0 = np.sqrt(1 + np.dot(sample.T, sample))
                # sample_hyp = [sample_x0, sample[0], sample[1]]

                # D1 = dhyp(prev_point_hyp, sample_hyp)
                # D2 = dhyp(sample_hyp, next_point_hyp)

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

# Straight line approximation
def compute_distance(x, y, Q):
    x = x[1:]
    y = y[1:]
    def integrand(t, x, y, Q):
        Pth = (1 - t) * x + y * t
        Dff = y-x
        ip = lambda x, y : np.matmul(Dff.T, np.matmul(Q.T, np.matmul(Q, Dff)))
        nm = lambda x, y : np.matmul(Pth.T, Dff)
        dn = lambda x, y : 1 + np.matmul(Pth.T, Pth)

        # print (ip(x, y) + (nm(x,y) ** 2 / dn(x, y)))
        return np.sqrt( ip(x, y) - (nm(x,y)**2 / dn(x, y)) )

    Distance = quad(integrand, 0, 1, args=(x, y, Q))
    return Distance[0]

# def mfd_dist(x, y):
#     return dhyp(x, y)

def mmc_Loss(Q, reg, mfd_dist):
    Q = Q.reshape(DIMENSION, DIMENSION)
    total = 0
    for edge in S:
        Qx = np.matmul(Q, edge[0][1:])
        Qy = np.matmul(Q, edge[1][1:])
        dQ = mfd_dist(Qx, Qy)
        total += (1 - reg) * dQ

    for edge in D:
        Qx = np.matmul(Q, edge[0][1:])
        Qy = np.matmul(Q, edge[1][1:])
        dQ = mfd_dist(Qx, Qy)
        total -= reg * dQ

    return total


def sim(x, Q, r, k):
    neighbors = []
    for edge in S:
        w = x == edge[0][1:]
        v = x == edge[1][1:]
        # w = x == edge[0]
        # v = x == edge[1]
        Qx = np.matmul(Q, x)
        if w.all():
            Qy = np.matmul(Q, edge[1][1:])
            # Qy = np.matmul(Q, edge[1])
            if np.linalg.norm(Qx - Qy)**2 < r**2:
                neighbors.append(Qy)
        elif v.all():
            Qy = np.matmul(Q, edge[0][1:])
            # Qy = np.matmul(Q, edge[0])
            if np.linalg.norm(Qx - Qy)**2 < r**2:
                neighbors.append(Qy)
        else:
            pass

    return neighbors

def impostor(x, Q, r):
    impostors = []
    for edge in D:
        w = x == edge[0][1:]
        v = x == edge[1][1:]
        # w = x == edge[0]
        # v = x == edge[1]
        Qx = np.matmul(Q, x)
        if w.all():
            Qy = np.matmul(Q, edge[1][1:])
            # Qy = np.matmul(Q, edge[1])
            if np.linalg.norm(Qx - Qy)**2 < r**2:
                impostors.append(Qy)
        elif v.all():
            Qy = np.matmul(Q, edge[0][1:])
            # Qy = np.matmul(Q, edge[1])
            if np.linalg.norm(Qx - Qy)**2 < r**2:
                impostors.append(Qy)
        else:
            pass

    return impostors

def mfd(x):
    return np.concatenate(([np.sqrt(1 + np.matmul(x.T, x))], x))

def lmnn_Loss(Q, radius, k, reg, mfd_dist):
    Q = Q.reshape(DIMENSION, DIMENSION)
    print(Q)
    total = 0
    for x in B:
        x = x[1:]
        Qx = np.matmul(Q, x)
        Qx = mfd(Qx)
        for Qy in sim(x, Q, radius, k):
            Qy = mfd(Qy)
            total += (1 - reg) * mfd_dist(Qx, Qy)

    for x in B:
        x = x[1:]
        Qx = np.matmul(Q, x)
        mQx = mfd(Qx)
        for Qy in sim(x, Q, radius, k):
            mQy = mfd(Qy)
            for Qz in impostor(x, Q, radius):
                mQz = mfd(Qz)
                if np.linalg.norm(Qx - Qz)**2 < np.linalg.norm(Qx - Qy)**2:
                    total += reg * (1 + mfd_dist(mQx, mQy) - mfd_dist(mQx, mQz))

    return total
