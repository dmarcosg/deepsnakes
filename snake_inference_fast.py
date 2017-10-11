import numpy as np
import scipy

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
    s = 10 #Number of subdivision per edge to compute the balloon force



    # Explicit time stepping for image energy minimization:

    snake_hist = []
    for step in range(step_n):
        u = np.int16(snake_u)
        v = np.int16(snake_v)
        a = alpha[u,v].squeeze()
        b = beta[u,v].squeeze()
        fu = Fu[u,v]
        fv = Fv[u,v]

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
        u_interps = np.int16(snake_u + np.arange(s)*(snake_u[np.roll(np.arange(L), -1)]-snake_u)/s)
        v_interps = np.int16(snake_v + np.arange(s)*(snake_v[np.roll(np.arange(L), -1)]-snake_v)/s)
        kappa_collection = kappa[u_interps,v_interps]


        # Get the derivative of the balloon energy
        js = np.arange(1, s + 1)
        s2 = 1 / (s * s)
        int_ends_u_next = s2 * (snake_u[np.roll(np.arange(L), -1)] - snake_u)  # snake_u[next_i] - snake_u[i]
        int_ends_u_prev = s2 * (snake_u[np.roll(np.arange(L), 1)] - snake_u)  # snake_u[prev_i] - snake_u[i]
        int_ends_v_next = s2 * (snake_v[np.roll(np.arange(L), -1)] - snake_v)  # snake_v[next_i] - snake_v[i]
        int_ends_v_prev = s2 * (snake_v[np.roll(np.arange(L), 1)] - snake_v)  # snake_v[prev_i] - snake_v[i]
        # contribution from the i+1 triangles to dE/du
        dEb_du = np.sum(js*kappa_collection[:, np.arange(s - 1, -1, -1)],axis=1) * int_ends_v_next.squeeze()
        dEb_du -= np.sum(js*kappa_collection[:, js - 1],axis=1) * int_ends_v_prev.squeeze()
        dEb_dv = -np.sum(js * kappa_collection[np.roll(np.arange(L), 1),:][:, np.arange(s - 1, -1, -1)], axis=1) * int_ends_u_next.squeeze()
        dEb_dv += np.sum(js * kappa_collection[np.roll(np.arange(L), 1),:][:, js - 1], axis=1) * int_ends_u_prev.squeeze()


        # Movements are capped to max_px_move per iteration:
        # duB = -max_px_move*np.tanh(2*np.matmul(B/np.square(delta_s),snake_u)/max_px_move)
        # duA = -max_px_move*np.tanh(2*np.matmul(A/delta_s,snake_u)/max_px_move)
        # dvB = -max_px_move*np.tanh(2 * np.matmul(B / np.square(delta_s), snake_v)/max_px_move)
        # dvA = -max_px_move*np.tanh(2 * np.matmul(A / delta_s, snake_v)/max_px_move)
        # duEb = max_px_move*np.tanh(dEb_du.reshape([L,1])/max_px_move)
        # dvEb = max_px_move * np.tanh(dEb_dv.reshape([L, 1]) / max_px_move)
        # duf = -max_px_move * np.tanh(fu / max_px_move)
        # dvf = -max_px_move * np.tanh(fv / max_px_move)
        # du = gamma*max_px_move * np.tanh((duB+duA+duEb+duf)/ max_px_move)*0.5 + du*0.5
        # dv = gamma*max_px_move * np.tanh((dvB + dvA + dvEb + dvf) / max_px_move) * 0.5 + dv * 0.5

        #du = -max_px_move*np.tanh( (fu - dEb_du.reshape([L,1]) + 2*np.matmul(A/delta_s+B/np.square(delta_s),snake_u))*gamma/max_px_move )*0.5 + du*0.5
        #dv = -max_px_move*np.tanh( (fv - dEb_dv.reshape([L,1]) + 2*np.matmul(A/delta_s+B/np.square(delta_s),snake_v))*gamma/max_px_move )*0.5 + dv*0.5

        du = -max_px_move*np.tanh( (fu - dEb_du.reshape([L,1]) )*gamma/max_px_move )*0.5 + du*0.5
        dv = -max_px_move*np.tanh( (fv - dEb_dv.reshape([L,1]) )*gamma/max_px_move )*0.5 + dv*0.5

        #snake_u += du
        #snake_v += dv
        #snake_u -= max_px_move*np.tanh(2 * np.matmul(A / delta_s + B / np.square(delta_s), snake_u)*gamma/max_px_move )
        #snake_v -= max_px_move*np.tanh(2 * np.matmul(A / delta_s + B / np.square(delta_s), snake_v)*gamma/max_px_move )
        snake_u = np.matmul( np.linalg.inv(np.eye(L,L) + 2*gamma*( A/delta_s+B/np.square(delta_s))),snake_u + gamma*du)
        snake_v = np.matmul( np.linalg.inv(np.eye(L,L) + 2*gamma*( A/delta_s+B/np.square(delta_s))),snake_v + gamma * dv)
        snake_u = np.minimum(snake_u, np.float32(M)-1)
        snake_v = np.minimum(snake_v, np.float32(N)-1)
        snake_u = np.maximum(snake_u, 1)
        snake_v = np.maximum(snake_v, 1)
        snake_hist.append(np.array([snake_u[:, 0], snake_v[:, 0]]).T)

    return snake_u,snake_v,du,dv,snake_hist
