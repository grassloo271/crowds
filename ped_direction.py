import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

class params:
    dt: float = 0.1
    V_0: float = 2.1
    del_t: float = 1
    sigma: float = 0.3

def semi_implicit_euler_hel(x: jnp.ndarray, speed:jnp.ndarray, theta:jnp.ndarray, a:jnp.ndarray, alpha: jnp.ndarray, p: params):
    a = jnp.linalg.norm(jnp.sum(a, axis=0), axis=-1)
    alpha = jnp.linalg.norm(jnp.sum(alpha, axis=0), axis=-1)
    speed_next = speed + a * p.dt 
    theta_next = theta + alpha * p.dt
    directions = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    x_next = x + p.dt * speed[:,None] * directions
    return x_next, speed_next, theta_next

def generate_data(x0, speed0, theta0, T, p, step_destination):
    def scan_body(carry, _ ):
        return step_destination(carry, p)
    
    (xT, vT, thetaT), (xs, vs, thetas) = jax.lax.scan(scan_body, (x0, speed0, theta0), jnp.zeros(T))
    return xs

#----------------------------Helbing model------------------------------- 

def project(x: jnp.array, theta: jnp.array):
    """ 
    >> x = project(jnp.array([[[2,0],[1,0]],[[0,1],[0,3]]]), jnp.array([0,jnp.pi]))
    """
    v_dir = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    x_parallel = jnp.sum(v_dir * x, axis = -1, keepdims=True) * v_dir
    x_perp = x - x_parallel
    return x_parallel, x_perp

def hel_force(x: jnp.array, speed: jnp.array, theta: jnp.array, p: params):
    """
    >> print(hel_force(jnp.array([[0,0], [2,0]]), speed=jnp.array([1,0.5]), theta=jnp.array([0, jnp.pi]), p=params))
    >> z=(hel_force(jnp.array([[0,0], [2,0], [3,0]]), speed=jnp.array([1,0.5, 1]), theta=jnp.array([0, jnp.pi, 0.75 * jnp.pi]), p=params))
    """
    eps = 1e-6
    diff = x[:, None, :] - x[None, :, :]
    
    D = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    
    vel = speed[:, None] * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1) 
    
    a1 = diff - vel[None, :, :] * p.del_t
    a1_norm = jnp.linalg.norm(a1, axis=-1, keepdims=True)
    b = 0.5 * jnp.sqrt((jnp.reshape(D + a1_norm, [x.shape[0], x.shape[0]])) ** 2 - (p.del_t * speed[None, :] )** 2)

    coeff = 1 / p.sigma * p.V_0 * jnp.exp(-b / p.sigma) / (b + jnp.eye(x.shape[0]))

    diff_un = diff / (D + eps)
    a1_un = a1 / (a1_norm + eps)

    mask = ~jnp.eye(x.shape[0], dtype=bool)
    direction = jnp.where(mask[:,:,None], a1_un + diff_un, jnp.zeros([x.shape[0], x.shape[0],2]))
    return coeff[:,:,None] * direction

def step_destination_hel(carry, p: params):
    x_next, speed_next, theta_next = carry
    F_parallel, F_perp = project(hel_force(x_next, speed_next, theta_next, p), theta_next)
    x_next, speed_next, theta_next = semi_implicit_euler_hel(x_next, speed_next, theta_next, F_parallel, F_perp, p)
    return (x_next, speed_next, theta_next), (x_next, speed_next, theta_next)

y = generate_data(x0=jnp.array([[0.0,0.0], [2.0,0.0]]), speed0=jnp.array([1.0,0.5]), theta0=jnp.array([1.0, jnp.pi]), T=100, p=params, step_destination=step_destination_hel)
print(y)
plt.plot(y[:5, 0], y[:5, 1])
plt.show()

# x0 = jnp.array([[0.0,0.0], [2.0,0.0]])
# speed0=jnp.array([1.0,0.5])
# theta0=jnp.array([0.0, jnp.pi])
# z = hel_force(x=x0, speed=speed0, theta=theta0, p = params)
# a, b = project(z, theta0)
# print(f"{a=}")
# c = jnp.sum(a, axis=0)
# d = jnp.linalg.norm(c, axis=-1)
# print(f"{c=}")
# print(f"{d=}")