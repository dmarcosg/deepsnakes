import numpy as np
from scipy import interpolate


def active_contour_step(Fu, Fv, du, dv, snake_u, snake_v, alpha, beta,
                    kappa, gamma,max_px_move, delta_s):
    """"" Perform one step in the minimization of the snake energy.
    Parameters
    ---------
    Fu, Fv: a MxN numpy arrays with the force fields in u and v
    du, dv: Lx1 numpy arrays with the previous steps (for momentum)
    snake_u, snake_v: Lx1 numpy arrays with the current snake
    alpha, beta: MxN numpy arays with the penalizations
    gamma: time step
    max_px_move: cap to the final step
    delta_s: desired distance between nodes
    Returns
    ----
    snake_u, snake_v: Lx1 numpy arrays with the new snake
    du, dv: Lx1 numpy arrays with the current steps (for momentum)
    """
    L = snake_u.shape[0]
    M = Fu.shape[0]
    N = Fu.shape[1]
    u = np.int32(np.round(snake_u))
    v = np.int32(np.round(snake_v))

    # Explicit time stepping for image energy minimization:

    a = []
    b = []
    k = []
    fu = []
    fv = []

    for i in range(L):
        a.append(alpha[u[i,0],v[i,0]])
        b.append(beta[u[i,0], v[i,0]])
        k.append(kappa[u[i, 0], v[i, 0]])
        fu.append(Fu[u[i,0], v[i,0]])
        fv.append(Fv[u[i,0], v[i,0]])
    a = np.stack(a)
    b = np.stack(b)
    k = np.stack(k).reshape([L,1])
    fu = np.reshape(np.stack(fu),u.shape)
    fv = np.reshape(np.stack(fv),v.shape)
    am1 = np.concatenate([a[L-1:L],a[0:L-1]],axis=0)
    a0d0 = np.diag(a)
    am1d0 = np.diag(am1)
    a0d1 = np.concatenate([a0d0[0:L,L-1:L], a0d0[0:L,0:L-1]], axis=1)
    am1dm1 = np.concatenate([am1d0[0:L, 1:L], am1d0[0:L, 0:1]], axis=1)

    bm1 = np.concatenate([b[L - 1:L], b[0:L - 1]],axis=0)
    b1 = np.concatenate([b[1:L], b[0:1]],axis=0)
    b0d0 = np.diag(b)
    bm1d0 = np.diag(bm1)
    b1d0 = np.diag(b1)
    b0dm1 = np.concatenate([b0d0[0:L, 1:L], b0d0[0:L, 0:1]], axis=1)
    b0d1 = np.concatenate([b0d0[0:L, L-1:L], b0d0[0:L, 0:L-1]], axis=1)
    bm1dm1 = np.concatenate([bm1d0[0:L, 1:L], bm1d0[0:L, 0:1]], axis=1)
    b1d1 = np.concatenate([b1d0[0:L, L - 1:L], b1d0[0:L, 0:L - 1]], axis=1)
    bm1dm2 = np.concatenate([bm1d0[0:L, 2:L], bm1d0[0:L, 0:2]], axis=1)
    b1d2 = np.concatenate([b1d0[0:L, L - 2:L], b1d0[0:L, 0:L - 2]], axis=1)


    A = -am1dm1  + (a0d0 + am1d0) - a0d1
    B = bm1dm2 - 2*(bm1dm1+b0dm1) + (bm1d0+4*b0d0+b1d0) - 2*(b0d1+b1d1) + b1d2

    n_u = np.concatenate([snake_v[1:L],snake_v[0:1]],axis=0)\
            - np.concatenate([snake_v[L-1:L],snake_v[0:L-1]],axis=0)
    n_v = np.concatenate([snake_u[L-1:L],snake_u[0:L-1]],axis=0)\
            - np.concatenate([snake_u[1:L],snake_u[0:1]],axis=0)
    norm = np.sqrt(np.power(n_u,2)+np.power(n_v,2))
    n_u = np.divide(n_u, norm)
    n_v = np.divide(n_v, norm)

    # Movements are capped to max_px_move per iteration:
    du = -max_px_move*np.tanh( (fu + 2*np.matmul(A/delta_s+B/np.square(delta_s),snake_u))*gamma )*0.1 + du*0.9
    dv = -max_px_move*np.tanh( (fv + 2*np.matmul(A/delta_s+B/np.square(delta_s),snake_v))*gamma )*0.1 + dv*0.9

    du += np.multiply(k,n_u)
    dv += np.multiply(k,n_v)

    snake_u += du
    snake_v += dv
    snake_u = np.minimum(snake_u, np.float32(M)-1)
    snake_v = np.minimum(snake_v, np.float32(N)-1)
    snake_u = np.maximum(snake_u, 1)
    snake_v = np.maximum(snake_v, 1)

    return snake_u,snake_v,du,dv

def draw_poly(poly,values,im_shape,total_points):
    """ Returns a MxN (im_shape) array with values in the pixels crossed
    by the edges of the polygon (poly). total_points is the maximum number
    of pixels used for the linear interpolation.
    """
    u = poly[:,0]
    v = poly[:,1]
    if type(values) is int:
        values = np.ones(np.shape(u)) * values
    [tck, s] = interpolate.splprep([u, v], s=2, k=1, per=1)
    [xi, yi] = interpolate.splev(np.linspace(0, 1, total_points), tck)
    intp = interpolate.interp1d(s, values)
    intp_values = intp(np.linspace(0, 1, total_points))
    image = np.zeros(im_shape)
    for n in range(len(xi)):
        image[int(xi[n]), int(yi[n])] += intp_values[n]
    return image

def derivatives_poly(poly):
    """
    :param poly: the Lx2 polygon array [u,v]
    :return: der1, der1, Lx2 derivatives arrays
    """
    u = poly[:, 0]
    v = poly[:, 1]
    L = len(u)
    der1_mat = -np.roll(np.eye(L), -1, axis=1) + \
               np.roll(np.eye(L), -1, axis=0)  # first order derivative, central difference
    der2_mat = np.roll(np.eye(L), -1, axis=0) + \
               np.roll(np.eye(L), -1, axis=1) - \
               2 * np.eye(L)  # second order derivative, central difference
    der1 = np.sqrt(np.power(np.matmul(der1_mat, u), 2) + \
                   np.power(np.matmul(der1_mat, v), 2))
    der2 = np.sqrt(np.power(np.matmul(der2_mat, u), 2) + \
                   np.power(np.matmul(der2_mat, v), 2))
    return der1,der2

def active_countour_gradients(snake,im_shape):
    L = snake.shape[0]
    der1_mat = -np.roll(np.eye(L), -1, axis=1) + \
               np.roll(np.eye(L), -1, axis=0)  # first order derivative, central difference
    der2_mat = np.roll(np.eye(L), -1, axis=0) + \
               np.roll(np.eye(L), -1, axis=1) - \
               2 * np.eye(L)  # second order derivative, central difference
    der1 = np.sqrt(np.power(np.matmul(der1_mat, snake[:, 0]), 2) + \
                   np.power(np.matmul(der1_mat, snake[:, 1]), 2))
    der2 = np.sqrt(np.power(np.matmul(der2_mat, snake[:, 0]), 2) + \
                   np.power(np.matmul(der2_mat, snake[:, 1]), 2))
    der0_img = np.zeros(im_shape)
    der1_img = np.zeros(im_shape)
    der2_img = np.zeros(im_shape)

    [tck, u] = interpolate.splprep([snake[:, 0], snake[:, 1]], s=2, k=1, per=1)
    [xi, yi] = interpolate.splev(np.linspace(0, 1, 200), tck)

    intp_der1 = interpolate.interp1d(u, der1)
    intp_der2 = interpolate.interp1d(u, der2)
    vals_der1 = intp_der1(np.linspace(0, 1, 200))
    vals_der2 = intp_der2(np.linspace(0, 1, 200))
    for n in range(len(xi)):
        print(n)
        der0_img[int(xi[n]), int(yi[n])] = 1
        der1_img[int(xi[n]), int(yi[n])] = vals_der1[n]
        der2_img[int(xi[n]), int(yi[n])] = vals_der2[n]

    gradients = np.zeros([im_shape[0], im_shape[1], 3])
    gradients[:, :, 0] = der0_img
    gradients[:, :, 1] = der1_img
    gradients[:, :, 2] = der2_img
    return gradients