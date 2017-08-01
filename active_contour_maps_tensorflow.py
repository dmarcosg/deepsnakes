import tensorflow as tf


def active_contour_step(Fu, Fv, du, dv, snake_u, snake_v, alpha, beta,kappa,
                    gamma,max_px_move, delta_s):

    L = snake_u.shape[0]
    M = Fu.shape[0]
    N = Fu.shape[1]
    u = tf.cast(tf.round(snake_u),tf.int32)
    v = tf.cast(tf.round(snake_v), tf.int32)


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
    a = tf.stack(a)
    b = tf.stack(b)
    k = tf.reshape(tf.stack(k),u.shape)
    fu = tf.reshape(tf.stack(fu),u.shape)
    fv = tf.reshape(tf.stack(fv),v.shape)
    am1 = tf.concat([a[L-1:L],a[0:L-1]],0)
    a0d0 = tf.diag(a)
    am1d0 = tf.diag(am1)
    a0d1 = tf.concat([a0d0[0:L,L-1:L], a0d0[0:L,0:L-1]], 1)
    am1dm1 = tf.concat([am1d0[0:L, 1:L], am1d0[0:L, 0:1]], 1)

    bm1 = tf.concat([b[L - 1:L], b[0:L - 1]],0)
    b1 = tf.concat([b[1:L], b[0:1]],0)
    b0d0 = tf.diag(b)
    bm1d0 = tf.diag(bm1)
    b1d0 = tf.diag(b1)
    b0dm1 = tf.concat([b0d0[0:L, 1:L], b0d0[0:L, 0:1]], 1)
    b0d1 = tf.concat([b0d0[0:L, L-1:L], b0d0[0:L, 0:L-1]], 1)
    bm1dm1 = tf.concat([bm1d0[0:L, 1:L], bm1d0[0:L, 0:1]], 1)
    b1d1 = tf.concat([b1d0[0:L, L - 1:L], b1d0[0:L, 0:L - 1]], 1)
    bm1dm2 = tf.concat([bm1d0[0:L, 2:L], bm1d0[0:L, 0:2]], 1)
    b1d2 = tf.concat([b1d0[0:L, L - 2:L], b1d0[0:L, 0:L - 2]], 1)


    A = -am1dm1  + (a0d0 + am1d0) - a0d1
    B = bm1dm2 - 2*(bm1dm1+b0dm1) + (bm1d0+4*b0d0+b1d0) - 2*(b0d1+b1d1) + b1d2

    n_u = tf.concat([snake_v[1:L], snake_v[0:1]], 0) \
          - tf.concat([snake_v[L - 1:L], snake_v[0:L - 1]], 0)
    n_v = tf.concat([snake_u[L - 1:L], snake_u[0:L - 1]], 0) \
          - tf.concat([snake_u[1:L], snake_u[0:1]], 0)
    norm = tf.sqrt(tf.pow(n_u, 2) + tf.pow(n_v, 2))
    n_u = tf.divide(n_u, norm)
    n_v = tf.divide(n_v, norm)

    # Movements are capped to max_px_move per iteration:
    du = -max_px_move*tf.tanh( (fu + 2*tf.matmul(A/delta_s+B/tf.square(delta_s),snake_u))*gamma )*0.1 + du*0.9
    dv = -max_px_move*tf.tanh( (fv + 2*tf.matmul(A/delta_s+B/tf.square(delta_s),snake_v))*gamma )*0.1 + dv*0.9

    du += tf.multiply(k, n_u)
    dv += tf.multiply(k, n_v)

    snake_u += du
    snake_v += dv
    snake_u = tf.minimum(snake_u, tf.cast(M,tf.float32)-1)
    snake_v = tf.minimum(snake_v, tf.cast(N,tf.float32)-1)
    snake_u = tf.maximum(snake_u, 1)
    snake_v = tf.maximum(snake_v, 1)

    return snake_u,snake_v,du,dv

