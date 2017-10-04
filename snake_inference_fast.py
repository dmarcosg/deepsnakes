import numpy as np

def active_contour_step(step_n, Fu, Fv, du, dv,
                    snake_u, snake_v, alpha, beta,
                    kappa, gamma, max_px_move, delta_s):
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
    s = 2 #Number of subdivision per edge to compute the balloon force
    kappa_collection = np.zeros((L,s))
    dEb_du = np.zeros((L,1))
    dEb_dv = np.zeros((L,1))
    u = np.int32(np.round(snake_u))
    v = np.int32(np.round(snake_v))

    # Explicit time stepping for image energy minimization:

    a = np.zeros((L))
    b = np.zeros((L))
    fu = np.zeros((L))
    fv = np.zeros((L))
    snake_hist = []
    for step in range(step_n):
        for i in range(L):
            a[i] = alpha[u[i,0],v[i,0]]
            b[i] = beta[u[i,0], v[i,0]]
            fu[i] = Fu[u[i,0], v[i,0]]
            fv[i] = Fv[u[i,0], v[i,0]]

        fu = np.reshape(fu,(L,1))
        fv = np.reshape(fv,(L,1))
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

        #Get kappa values between nodes
        for i in range(L):
            next_i = i + 1
            if next_i == L:
                next_i = 0
            u_interp = np.int32(snake_u[i]+range(s)*(snake_u[next_i]-snake_u[i])/s)
            v_interp = np.int32(snake_v[i]+range(s)*(snake_v[next_i]-snake_v[i])/s)
            for j in range(s):
                kappa_collection[i,j] = kappa[u_interp[j], v_interp[j]]


        #Get the derivative of the balloon energy
        for i in range(L):
            next_i = i + 1
            prev_i = i - 1
            if next_i == L:
                next_i = 0
            if prev_i == -1:
                prev_i = L-1
            val = 0
            #contribution from the i+1 triangle to dE/du
            int_end = snake_v[next_i] - snake_v[i]
            dh = np.abs(int_end/s)
            for j in range(s):
                val += np.sign(int_end)*(j+1)/s * kappa_collection[i,s-j-1] * dh
            #contribution from the i-1 triangle to dE/du
            int_end = snake_v[prev_i] - snake_v[i]
            dh = np.abs(int_end / s)
            for j in range(s):
                val += -np.sign(int_end)*(j+1)/s * kappa_collection[i,j] * dh
            dEb_du[i] = val

            val = 0
            # contribution from the i+1 triangle to dE/dv
            int_end = snake_u[next_i] - snake_u[i]
            dh = np.abs(int_end / s)
            for j in range(s):
                val += -np.sign(int_end)*(j+1) / s  * kappa_collection[prev_i,s-j-1] * dh
            # contribution from the i-1 triangle to dE/dv
            int_end = snake_u[prev_i] - snake_u[i]
            dh = np.abs(int_end / s)
            for j in range(s):
                val += np.sign(int_end)*(j+1) / s  * kappa_collection[prev_i,j] * dh
            dEb_dv[i] = val




        # Movements are capped to max_px_move per iteration:
        du = -max_px_move*np.tanh( (fu - dEb_du + 2*np.matmul(A/delta_s+B/np.square(delta_s),snake_u))*gamma )*0.1 + du*0.9
        dv = -max_px_move*np.tanh( (fv - dEb_dv + 2*np.matmul(A/delta_s+B/np.square(delta_s),snake_v))*gamma )*0.1 + dv*0.9

        #du += np.multiply(k,n_u)
        #dv += np.multiply(k,n_v)
        #du += dEb_du
        #dv += dEb_dv

        snake_u += du
        snake_v += dv
        snake_u = np.minimum(snake_u, np.float32(M)-1)
        snake_v = np.minimum(snake_v, np.float32(N)-1)
        snake_u = np.maximum(snake_u, 1)
        snake_v = np.maximum(snake_v, 1)
        snake_u[L-1] = snake_u[0]
        snake_v[L - 1] = snake_v[0]
        snake_hist.append(np.array([snake_u[:, 0], snake_v[:, 0]]).T)

    return snake_u,snake_v,du,dv,snake_hist
