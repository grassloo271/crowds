import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import jax 

class params:
    tau = 1
    v_target_mag = 3
    A: float = 5.0
    B: float = 1.0
    d0: float = 0.5
    r_cut = 5

# ================================================saving data=========================

def people_to_npz(crowd, name):
    """
    Takes a list of people that you want to do something and
    puts them in a npz fiile for easier reading later on
    """
    gen_arr = [(ppl.x, ppl.v, ppl.color, ppl.goal, ppl.is_goal) for ppl in crowd]
    x, v, color, goal, is_goal = zip(*gen_arr)
    np.savez(
        name,
        x=np.asarray(x),
        v=np.asarray(v),
        goal=np.asarray(goal),
        is_goal = np.asarray(is_goal),
        color=np.asarray(color),
    )
# ====================================rendering data==================================

def render_animation_gif(
    npz_file_name,
    dt: float,
    fps: int = 30,
    tail_len: int = 400,  # number of past frames to show as trails
    s: int = 5,  # scatter point size
):
    """
    takes a npz file labeled like the readme and makes a gif out of it
    just specify dt
    """
    data = np.load(npz_file_name)

    positions = jnp.array(data["x"])
    coloration = data["color"]
    out_path_gif = str(npz_file_name) + ".gif"

    """
    Try to render an MP4 (ffmpeg). If ffmpeg is unavailable, fall back to GIF (Pillow).
    """
    pos = np.asarray(positions)  # [T,N,2] as numpy
    print(pos.shape)
    T, N, _ = pos.shape

    # Axis limits with a small margin
    xmin, xmax = pos[..., 0].min(), pos[..., 0].max()
    ymin, ymax = pos[..., 1].min(), pos[..., 1].max()
    margin_x = 0.1 * (xmax - xmin + 1e-6)
    margin_y = 0.1 * (ymax - ymin + 1e-6)
    xlim = (xmin - margin_x, xmax + margin_x)
    ylim = (ymin - margin_y, ymax + margin_y)

    # Colors: left group (0..num_left-1) vs right group
    colors = coloration
    #    colors = np.array(["tab:blue"] * num_left + ["tab:orange"] * num_right)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Two-Group Pedestrian Flow")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Scatter for current positions
    scat = ax.scatter([], [], s=s)

    # Optional trails: a set of Line2D objects per agent
    lines = [ax.plot([], [], lw=1, alpha=0.6, color=colors[i])[0] for i in range(N)]

    # Legend
    # from matplotlib.lines import Line2D
    # legend_elems = [
    #     Line2D([0], [0], marker='o', color='w', label='Left group', markerfacecolor='tab:blue', markersize=8),
    #     Line2D([0], [0], marker='o', color='w', label='Right group', markerfacecolor='tab:orange', markersize=8),
    # ]
    # ax.legend(handles=legend_elems, loc="upper right")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        for ln in lines:
            ln.set_data([], [])
        return [scat, *lines]

    def update(frame):
        # Current positions
        XY = pos[frame]  # [N,2]
        scat.set_offsets(XY)
        scat.set_color(colors)

        # Trails: show last `tail_len` positions (clamped at 0)
        t0 = max(0, frame - tail_len)
        trail = pos[t0 : frame + 1]  # [L,N,2]
        for i in range(N):
            lines[i].set_data(trail[:, i, 0], trail[:, i, 1])
        ax.set_title(f"Two-Group Pedestrian Flow  |  t={frame}  (dt={dt:.3f})")
        return [scat, *lines]

    interval_ms = int(1000 / fps)
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=T, interval=interval_ms, blit=True
    )

    # Try MP4 first (ffmpeg), else fallback to GIF (Pillow)
    try:
        from matplotlib.animation import PillowWriter

        anim.save(out_path_gif, writer=PillowWriter(fps=fps))
        print(f"Saved animation (GIF) to: {out_path_gif}")
    except Exception as e:
        print("Failed to save animation as GIF too:", e)

    plt.close(fig)

# =========================================projection====================================

def project(x: jnp.array, v: jnp.array):
    """
    projects a matrix of vectors in x onto a list of vectors.

    >> x = project(jnp.array([[[2,0],[1,0]],[[0,1],[0,3]]]), jnp.array([0,jnp.pi]))
    """
    eps = 1e-6
    v_dir = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)
    x_parallel = jnp.sum(v_dir[:,None,:] * x, axis=-1, keepdims=True) * v_dir
    x_perp = x - x_parallel
    return x_parallel, x_perp

def absolute_to_projected(x: jnp.array, v: jnp.array):
    """
    takes the npz data and turns it into the x_par, x_perp, v_par, v_perp
    """
    x_diff = x[:, None, :] - x[None, :, :]
    v_diff = v[:, None, :] - v[None, :, :]

    v_par, v_perp = project(v_diff, v)
    x_par, x_perp = project(x_diff, x)
    return jnp.sum(x_par, axis=-1), jnp.sum(x_perp, axis = -1), jnp.sum(v_par, axis=-1), jnp.sum(v_perp, axis=-1)

def projected_to_absolute(f_par, f_perp, v):
    """
    turns the projected forces into absolutes to easier deal with in the euler.
    """
    eps = 1e-6
    v_dir = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)
    rot = jnp.array([[0, -1], [1, 0]])
    v_perp = (rot @ v_dir.T).T
    f_pa = jnp.sum(f_par[:, :, None] * v_dir[None, :, :], axis=0)
    f_per = jnp.sum(f_perp[:, :, None] * v_perp[None, :, :], axis=0)
    return f_pa + f_per

# ================================euler initegration====================================

def semi_implicit_euler(x, v, a, dt):
    v = v + a * dt
    x = x + v * dt
    return x, v

# =======================================solve=========================================

def solve(init_cond, f_par, f_perp, f_attr, dt, time):
    """
    takes in a function f_par, f_perp and plugs in the new things
    """
    x0, v0 = init_cond
    
    def step(carry, par, perp, dt):
        x, v = carry
        x_par, x_perp, v_par, v_perp = absolute_to_projected(x, v)
        F_par = par(x_par, x_perp, v_par, v_perp)
        F_perp = perp(x_par, x_perp, v_par, v_perp)

        F_attr = f_attr(x, v)

        F_tot = projected_to_absolute(F_par, F_perp, v) + F_attr
        x_next, v_next = semi_implicit_euler(x, v, F_tot, dt)
        return (x_next, v_next), (x_next, v_next)
    
    def scan_body(carry, _):
        return step(carry, f_par, f_perp, dt)
    
    (xT, vT), (xs, vs) = jax.lax.scan(scan_body, (x0, v0), jnp.zeros(time))

    positions = jnp.concatenate([x0[None, ...], xs], axis=0)
    velocities = jnp.concatenate([v0[None, ...], vs], axis=0)
    return positions, velocities

#==================================attractive force==============================

def f_attractive(x, v, goal, is_goal_point, p:params):

    def point_attraction(x, v, goal, p):
        eps = 1e-6
        diff = goal - x
        norms = jnp.linalg.norm(diff, axis=1, keepdims=True)
        v_target = p.v_target_mag * diff / (norms+eps)
        return (v_target - v) / p.tau
    
    def vel_attraction(x, v, goal, p):
        return (goal - v) / p.tau
    
    point_goal = is_goal_point.astype(int)
    vel_goal = (~is_goal_point).astype(int)
    print(point_goal[:,None] * point_attraction(x, v, goal, p) + vel_goal[:,None] * vel_attraction(x, v, goal, p))

    return point_goal[:,None] * point_attraction(x, v, goal, p) + vel_goal[:,None] * vel_attraction(x, v, goal, p)

#=====================================forces======================================

def f_test_par(*args):
    return 0*jnp.ones((1,80))

def f_test_perp(*args):
    return 0*jnp.ones((1,80))

#====================================exponential====================================

def f_exponential_repulsion(x: jnp.ndarray,  p: params) -> jnp.ndarray:
    N = x.shape[0]
    diff = x[:, None, :] - x[None, :, :]      # [N, N, 2]
    d2 = jnp.sum(diff * diff, axis=-1)        # [N, N]
    eps = 1e-6
    d = jnp.sqrt(d2 + eps)                    # [N, N]
    n_ij = diff / d[..., None]                # [N, N, 2]
    eye = jnp.eye(N, dtype=jnp.float32)
    mag = p.A * jnp.exp((p.d0 - d) / p.B) * (1.0 - eye)
    if p.r_cut > 0.0:
        mag = mag * (d <= p.r_cut)
    F = jnp.sum(mag[..., None] * n_ij, axis=1)  # [N, 2]
    return F

#==============================time_to_collision====================================

def f_time_to_collision(x_par, x_perp, v_par, v_perp):
    eps = 1e-6
    a = jnp.sqrt(v_par * v_par + v_perp * v_perp)
    b = jnp.sqrt(x_par * x_par + x_perp * x_perp)
    c = (x_par * v_par + x_perp * v_perp) / ((a + eps)* (b + eps))
    print(c)

if __name__ == "__main__":
    # x,y= project(jnp.array([[[2,0],[1,0]],[[0,1],[0,3]]]), jnp.array([[0,1],[1,0]]))

    crowd = np.load("oop_init.npz")

    init_cond = (crowd["x"], crowd["v"])
    x, v = init_cond

    x_par, x_perp, v_par, v_perp = absolute_to_projected(x, v)
    
    f_time_to_collision(x_par, x_perp, v_par, v_perp)
else:

    f_attr = lambda x, v: f_attractive(x, v, crowd["goal"], crowd["is_goal"], params)
    f_exp = lambda x, v: f_exponential_repulsion(x, params)

    f_tot = lambda x, v: f_attractive(x, v, crowd["goal"], crowd["is_goal"], params) + f_exponential_repulsion(x, params)

    x_new, v_new = solve(init_cond, f_test_par, f_test_perp, f_tot, 0.1, 500)
   
    np.savez(
        "oop.npz",
        x=np.asarray(x_new),
        v=np.asarray(v_new),
        goal=crowd["goal"],
        color = crowd["color"]
    )

    render_animation_gif("oop.npz", 0.1, tail_len=100)
