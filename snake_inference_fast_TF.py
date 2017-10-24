import tensorflow as tf


def active_contour_step(Fu, Fv, du, dv, snake_u, snake_v, alpha, beta,kappa,
                    gamma,max_px_move, delta_s):

    L = snake_u.shape[0]
    M = Fu.shape[0]
    N = Fu.shape[1]
    u = tf.cast(tf.round(snake_u),tf.int32)
    v = tf.cast(tf.round(snake_v), tf.int32)


    # Explicit time stepping for image energy minimization:

    fu = tf.gather(tf.reshape(Fu, tf.TensorShape([M * N])), u*M + v)
    fv = tf.gather(tf.reshape(Fv, tf.TensorShape([M * N])), u * M + v)
    a = tf.gather(tf.reshape(alpha, tf.TensorShape([M * N])), u * M + v)
    b = tf.gather(tf.reshape(beta, tf.TensorShape([M * N])), u * M + v)
    a = tf.squeeze(a)
    b = tf.squeeze(b)
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

    # Get kappa values between nodes
    s = 10
    range_float = tf.cast(tf.range(s),tf.float32)
    snake_u1 = tf.concat([snake_u[L-1:L],snake_u[0:L-1]],0)
    snake_v1 = tf.concat([snake_v[L-1:L],snake_v[0:L-1]],0)
    snake_um1 = tf.concat([snake_u[1:L], snake_u[0:1]], 0)
    snake_vm1 = tf.concat([snake_v[1:L], snake_v[0:1]], 0)
    u_interps = tf.cast(snake_u+tf.round(tf.multiply(range_float ,(snake_um1 - snake_u)) / s), tf.int32)
    v_interps = tf.cast(snake_v+tf.round(tf.multiply(range_float ,(snake_vm1 - snake_v)) / s), tf.int32)
    kappa_collection = tf.gather(tf.reshape(kappa, tf.TensorShape([M * N])), u_interps*M  + v_interps)
    #kappa_collection = tf.reshape(kappa_collection,tf.TensorShape([L,s]))
    #kappa_collection = tf.Print(kappa_collection,[kappa_collection],summarize=1000)
    # Get the derivative of the balloon energy
    js = tf.cast(tf.range(1, s + 1),tf.float32)
    s2 = 1 / (s * s)
    int_ends_u_next = s2 * (snake_um1 - snake_u)  # snake_u[next_i] - snake_u[i]
    int_ends_u_prev = s2 * (snake_u1 - snake_u)  # snake_u[prev_i] - snake_u[i]
    int_ends_v_next = s2 * (snake_vm1 - snake_v)  # snake_v[next_i] - snake_v[i]
    int_ends_v_prev = s2 * (snake_v1 - snake_v)  # snake_v[prev_i] - snake_v[i]
    # contribution from the i+1 triangles to dE/du

    dEb_du = tf.multiply(tf.reduce_sum(tf.multiply(js,
              tf.gather(kappa_collection,
              tf.range(s-1,-1,delta=-1),axis=1)),axis=1),
              tf.squeeze(int_ends_v_next))
    dEb_du -= tf.multiply(tf.reduce_sum(tf.multiply(js,
              kappa_collection), axis=1),
              tf.squeeze(int_ends_v_prev))

    dEb_dv = -tf.multiply(tf.reduce_sum(tf.multiply(js,
             tf.gather(tf.gather(kappa_collection,tf.concat([tf.range(L-1,L),tf.range(L-1)],0),axis=0),
             tf.range(s - 1, -1, delta=-1), axis=1)), axis=1),
             tf.squeeze(int_ends_u_next))
    dEb_dv += tf.multiply(tf.reduce_sum(tf.multiply(js,
              tf.gather(kappa_collection,
              tf.concat([tf.range(L-1, L ), tf.range(L - 1)],0), axis=0),), axis=1),
              tf.squeeze(int_ends_u_prev))
    #dEb_du = np.sum(js * kappa_collection[:, np.arange(s - 1, -1, -1)], axis=1) * int_ends_v_next.squeeze()
    #dEb_du -= np.sum(js * kappa_collection[:, js - 1], axis=1) * int_ends_v_prev.squeeze()
    #dEb_dv = -np.sum(js * kappa_collection[np.roll(np.arange(L), 1), :][:, np.arange(s - 1, -1, -1)],
    #                 axis=1) * int_ends_u_next.squeeze()
    #dEb_dv += np.sum(js * kappa_collection[np.roll(np.arange(L), 1), :][:, js - 1], axis=1) * int_ends_u_prev.squeeze()




    # Movements are capped to max_px_move per iteration:
    #du = -max_px_move*tf.tanh( (fu - tf.reshape(dEb_du,fu.shape) + 2*tf.matmul(A/delta_s+B/tf.square(delta_s),snake_u))*gamma )*0.5 + du*0.5
    #dv = -max_px_move*tf.tanh( (fv - tf.reshape(dEb_dv,fv.shape) + 2*tf.matmul(A/delta_s+B/tf.square(delta_s),snake_v))*gamma )*0.5 + dv*0.5
    du = -max_px_move * tf.tanh((fu  - tf.reshape(dEb_du,fu.shape))*gamma )*0.5 + du*0.5
    dv = -max_px_move*tf.tanh( (fv  - tf.reshape(dEb_dv,fv.shape))*gamma )*0.5 + dv*0.5
    snake_u = tf.matmul(tf.matrix_inverse(tf.eye(L._value) + 2*gamma*(A/delta_s + B/(delta_s*delta_s))), snake_u + gamma * du)
    snake_v = tf.matmul(tf.matrix_inverse(tf.eye(L._value) + 2 * gamma * (A / delta_s + B / (delta_s * delta_s))), snake_v + gamma * dv)

    #snake_u = np.matmul(np.linalg.inv(np.eye(L, L) + 2 * gamma * (A / delta_s + B / np.square(delta_s))),
    #                    snake_u + gamma * du)
    #snake_v = np.matmul(np.linalg.inv(np.eye(L, L) + 2 * gamma * (A / delta_s + B / np.square(delta_s))),
    #                    snake_v + gamma * dv)

    #snake_u += du
    #snake_v += dv
    snake_u = tf.minimum(snake_u, tf.cast(M,tf.float32)-1)
    snake_v = tf.minimum(snake_v, tf.cast(N,tf.float32)-1)
    snake_u = tf.maximum(snake_u, 1)
    snake_v = tf.maximum(snake_v, 1)

    return snake_u,snake_v,du,dv

