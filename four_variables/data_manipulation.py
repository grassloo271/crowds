import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import numpy as np
import jax 

class params:
    # tau = 0.5 #exponential
    tau=.3
    v_target_mag = 1
    # v_target_mag = .8 # exponential normal
    A: float = 0.5
    B: float = 0.5
    d0: float = 0.3
    r_cut = 5
    right_a = 0.1
    right_b = 0.1

# ================================================saving data=========================

def overlay_npz(file1, file2, out_name):
    """
    Overlays two npz crowd files (saved via people_to_npz) so they 
    render simultaneously. Concatenates along the agent axis (N).
    """
    a = np.load(file1)
    b = np.load(file2)

    # x and v are [T, N, 2] — pad shorter one to match T
    T_a = a["x"].shape[0]
    T_b = b["x"].shape[0]
    T = max(T_a, T_b)

    def pad_time(arr, T_target):
        """Pad a [T, N, ...] array along axis 0 by repeating last frame."""
        if arr.shape[0] < T_target:
            pad = np.repeat(arr[[-1]], T_target - arr.shape[0], axis=0)
            return np.concatenate([arr, pad], axis=0)
        return arr

    x = np.concatenate([pad_time(a["x"], T), pad_time(b["x"], T)], axis=1)
    #v = np.concatenate([pad_time(a["v"], T), pad_time(b["v"], T)], axis=1)

    # 1D arrays along N axis — simple concatenate
    color   = np.concatenate([a["color"],   b["color"]])
    # goal    = np.concatenate([a["goal"],    b["goal"]])
    # is_goal = np.concatenate([a["is_goal"], b["is_goal"]])
    # enter_t = np.concatenate([a["enter_t"], b["enter_t"]])

    np.savez(
        out_name,
        x=x,
        #v=v,
        color=color,
        # goal=goal,
        # is_goal=is_goal,
        # enter_t=enter_t,
    )
    print(f"Saved overlay to {out_name} — {x.shape[1]} total agents, {T} frames")

def people_to_npz(crowd, name):
    """
    Takes a list of people that you want to do something and
    puts them in a npz fiile for easier reading later on
    """
    gen_arr = [(ppl.x, ppl.v, ppl.color, ppl.goal, ppl.is_goal, ppl.enter_t) for ppl in crowd]
    x, v, color, goal, is_goal, enter_t = zip(*gen_arr)
    np.savez(
        name,
        x=np.asarray(x),
        v=np.asarray(v),
        goal=np.asarray(goal),
        is_goal = np.asarray(is_goal),
        color=np.asarray(color),
        enter_t = np.asarray(enter_t)
    )
# ====================================rendering data==================================

def render_animation_gif(
    npz_file_name,
    dt: float,
    enter_t=0,
    leave_t = 100, 
    fps: int = 30,
    tail_len: int = 400,  # number of past frames to show as trails
    s: int = 5,  # scatter point size
    xrange = (-100,100),
    yrange=(-100,100)
):
    print("HI.")
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
    T, N, _ = pos.shape
    if type(enter_t) is float or type(enter_t) is int:
        enter_t = jnp.zeros(N)

    if type(leave_t) in (int, float):
        leave_t = jnp.ones(N) * 1000

    # Axis limits with a small margin
    xl, xh = xrange
    yl, yh = yrange
    xmin, xmax =pos[..., 0][pos[..., 0] > xl].min(),  pos[..., 0][pos[..., 0] < xh].max()
    ymin, ymax =  pos[..., 1][pos[..., 1] > yl].min(),  pos[..., 1][pos[..., 1] < yh].max()
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

# replace the single scat with one per agent
    
    def init():
        scat.set_offsets(np.empty((0, 2)))
        for ln in lines:
            ln.set_data([], [])
        return [scat, *lines] 

    def update(frame):
        # Current positions
        active = np.array((enter_t <= frame )&(frame  <= leave_t))
        
        XY = np.where(active[:, None], pos[frame], np.nan)  # [N,2]
        scat.set_offsets(XY)
        scat.set_color(colors)

        # Trails: show last `tail_len` positions (clamped at 0)
        t0 = max(0, frame - tail_len)
        trail = pos[t0 : frame + 1]  # [L,N,2]
        for i in range(N):
            if active[i]:
                lines[i].set_data(trail[:, i, 0], trail[:, i, 1])
            else:
                lines[i].set_data([], [])  # hide trail for agents not yet entered
        ax.set_title(f"Two-Group Pedestrian Flow  |  t={frame}  (dt={dt:.3f})")
        return [scat, *lines]

    interval_ms = int(1000 / fps)
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=T, interval=interval_ms, blit=False
    )

    # Try MP4 first (ffmpeg), else fallback to GIF (Pillow)
    try:
        anim.save(out_path_gif.replace(".gif", ".mp4"), writer=FFMpegWriter(fps=fps))
        # anim.save(out_path_gif, writer=PillowWriter(fps=fps))
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
    
    x_par = jnp.sum(x*v_dir[None,:, :], axis=-1)
    x_parallel = jnp.sum(x*v_dir[None,:, :], axis=-1, keepdims=True)
    x_par_vec = x_parallel * v_dir
    x_perp = x - x_par_vec

    rot = jnp.array([[0, -1], [1, 0]])
    x_perp_real = (rot @ v_dir.T).T
    
    x_perp = jnp.sum(x_perp * x_perp_real[None, :, :], axis=-1)
    return x_par, x_perp

def absolute_to_projected(x: jnp.array, v: jnp.array):
    """
    takes the npz data and turns it into the x_par, x_perp, v_par, v_perp
    """
    x_diff = x[:, None, :] - x[None, :, :]
   
    v_diff = v[:, None, :] - v[None, :, :]

    v_par, v_perp = project(v_diff, v)
    x_par, x_perp = project(x_diff, v)
    
    return x_par, x_perp, v_par,v_perp

def projected_to_absolute(f_par, f_perp, v):
    """
    turns the projected forces into absolutes to easier deal with in the euler.
    """
    eps = 1e-6
    v_dir = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)

    rot = jnp.array([[0, -1], [1, 0]])
    v_perp = (rot @ v_dir.T).T
    f_pa = f_par[:,  None] * v_dir
    f_per = f_perp[:,  None] * v_perp
    return f_pa + f_per

# ================================euler initegration====================================
def semi_implicit_euler(x, v, a, dt):
    v = v + a * dt
    x = x + v * dt
    return x, v

# =======================================solve=========================================

def solve_for(init_cond, f_par, f_perp, dt, time, enter_t=0, leave_t = 0, f_attr=lambda x,v: 0):
    x, v = init_cond
    if type(enter_t) is float or type(enter_t) is int:
        enter_t = jnp.zeros(x.shape[0])

    if type(leave_t) in (int, float):
        leave_t = 1000 * jnp.ones(x.shape[0])

    xs, vs = [x], [v]
    
    for t in range(time):
        active = (enter_t<= t )& (t <= leave_t)
        
        x_par1, x_perp1, v_par, v_perp = absolute_to_projected(x,v)
        
        x_perp = jnp.where(active[:, None], x_perp1, 1000 * jnp.ones(x.shape[0]))

        x_par = jnp.where(active[:, None], x_par1, 1000* jnp.ones(x.shape[0]))

        F_par = f_par(x_par, x_perp, v_par, v_perp)
        
        F_perp = f_perp(x_par, x_perp, v_par, v_perp)

        F_tot = projected_to_absolute(F_par, F_perp, v)
        F_tot = F_tot + f_attr(x, v)
        F_tot = jnp.where(active[:, None], F_tot, 0)   
        
        v_mask = jnp.where(active[:, None], v, 0)
        x, v = semi_implicit_euler(x, v_mask, F_tot, dt)
        xs.append(x)
        vs.append(v)
    return xs, vs 


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

    return point_goal[:,None] * point_attraction(x, v, goal, p) + vel_goal[:,None] * vel_attraction(x, v, goal, p)

#=====================================forces======================================

def f_test_par(*args):
    return 0*jnp.ones(120)

def f_test_perp(*args):
    return 0*jnp.ones(120)

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

def f_exponential_repulsion_par(x_par, x_perp, p_par, p_perp, p = params):
    N = x_par.shape[0]
    dist = jnp.sqrt(x_par * x_par + x_perp * x_perp)
    eps = 1e-6
    eye = jnp.eye(N, dtype=jnp.float32)
    mag = p.A * jnp.exp((p.d0 - dist) / p.B) * (1.0 - eye)

    if p.r_cut > 0.0:
        mag = mag * (dist <= p.r_cut)

    F = x_par / (dist + eps) * mag 
    F = jnp.sum(F, axis = 0)
    return -F

def f_exponential_repulsion_perp(x_par, x_perp, v_par, v_perp, p= params):
    N = x_par.shape[0]
    dist = jnp.sqrt(x_par * x_par + x_perp * x_perp)
    eps = 1e-6
    eye = jnp.eye(N, dtype=jnp.float32)
    mag = p.A * jnp.exp((p.d0 - dist)/ p.B) * (1.0 - eye)
    if p.r_cut > 0.0:
        mag = mag * (dist <= p.r_cut)
    F = x_perp / (dist + eps)* mag 
    
    F = jnp.sum(F, axis = 0)
    return -F

#==============================left right==========================================

def exp_right_perp(x_par, x_perp, v_par, v_perp, p=params):
    N = x_par.shape[0]
    dist = jnp.sqrt(x_par * x_par + x_perp * x_perp)
    eps = 1e-6
    eye = jnp.eye(N, dtype=jnp.float32)
    mag = p.A * jnp.exp((p.d0 - dist)/ p.B) * (1.0 - eye)
    if p.r_cut > 0.0:
        mag = mag * (dist <= p.r_cut)
        F = (x_perp) / (dist + eps)* mag + p.right_a * 2 * jnp.exp(- dist ** 3 * 50 * p.right_b)
    
    F = jnp.sum(F, axis = 0)
    return -F

#=================================back front ============================================

def front_par(x_par, x_perp, p_par, p_perp, p = params):
    N = x_par.shape[0]
    mask = x_par > 0

    dist = jnp.sqrt(x_par * x_par + x_perp * x_perp)
    eps = 1e-6
    eye = jnp.eye(N, dtype=jnp.float32)
    mag = p.A * jnp.exp((p.d0 - dist) / p.B) * (1.0 - eye)

    mag = jnp.where(mask, mag, 0)

    if p.r_cut > 0.0:
        mag = mag * (dist <= p.r_cut)

    F = x_par / (dist + eps) * mag 
    F = jnp.sum(F, axis = 0)
    return -F

def front_perp(x_par, x_perp, v_par, v_perp, p=params):
    N = x_par.shape[0]
    dist = jnp.sqrt(x_par * x_par + x_perp * x_perp)
    mask = x_par > 0

    eps = 1e-6
    eye = jnp.eye(N, dtype=jnp.float32)
    mag = p.A * jnp.exp((p.d0 - dist)/ p.B) * (1.0 - eye)

    mag = jnp.where(mask, mag, 0)
    if p.r_cut > 0.0:
        mag = mag * (dist <= p.r_cut)
        F = (x_perp) / (dist + eps)* mag + p.right_a * 2 * jnp.exp(- dist ** 3 * 50 * p.right_b)
    
    F = jnp.sum(F, axis = 0)
    return -F

#==============================time_to_collision====================================

def f_time_to_collision(x_par, x_perp, v_par, v_perp):
    eps = 1e-6
    a = jnp.sqrt(v_par * v_par + v_perp * v_perp)
    b = jnp.sqrt(x_par * x_par + x_perp * x_perp)
    c = (x_par * v_par + x_perp * v_perp) / ((a + eps)* (b + eps))

if __name__ == "__main__":
    case = 1
    if case == 0:
        x,y= project(jnp.array([[[2,0],[1,0]],[[0,1],[0,3]]]), jnp.array([[0,1],[1,0]]))
        y = jnp.stack((jnp.arange(3),jnp.arange(3)), axis=-1)
        
        z = jnp.stack((jnp.arange(3), jnp.ones(3)), axis= -1) 
        
        x_par, x_perp, v_par, v_perp = absolute_to_projected(y, z)
        

        f_exponential_repulsion_par(x_par, x_perp, v_par, v_perp)
        print(projected_to_absolute(jnp.array([2,5]), jnp.array([1,1]), jnp.array([2,3])))
    elif case == 1:
        name = "s11_init"
        crowd = np.load(name + ".npz")

        init_cond = (crowd["x"], crowd["v"])
        x, v = init_cond
        x_par, x_perp, v_par, v_perp = absolute_to_projected(x, v)
        
        f_time_to_collision(x_par, x_perp, v_par, v_perp)

        params.tau=1
        f_attr = lambda x, v: f_attractive(x, v, crowd["goal"], crowd["is_goal"], params)
        # f_attr = lambda x, v:0
        # f_exp = lambda x, v: f_exponential_repulsion(x, params)

        f_tot = lambda x, v: f_attractive(x, v, crowd["goal"], crowd["is_goal"], params) + f_exponential_repulsion(x, params)
        # f_tot = lambda x, v: 0
        # x_new,v_new = solve(init_cond, f_test_par, f_test_perp, f_tot, 0.1, 100)
        print("here") 
        x_new, v_new = solve_for(init_cond, f_exponential_repulsion_par, f_exponential_repulsion_perp, 0.1, 160, enter_t=crowd["enter_t"], leave_t = crowd["leave_t"], f_attr = f_attr)
        # x_new, v_new = solve_for(init_cond, f_exponential_repulsion_par, exp_right_perp, 0.1, 100, f_attr)
        # x_new, v_new = solve_for(init_cond, f_exponential_repulsion_par, f_exponential_repulsion_perp, 0.1, 100, f_attr)
        
        save_name = name + "_run.npz"
        np.savez(
            save_name,
            x=np.asarray(x_new),
            v=np.asarray(v_new),
            goal=crowd["goal"],
            color = crowd["color"]
        )
# 
        # overlay_npz("s11.npz", "s11_init_run_front.npz", "combined_front.npz")
        # render_animation_gif("combined_front.npz", dt=0.1, tail_len=2)    
        render_animation_gif(save_name, 0.1, enter_t=crowd["enter_t"], tail_len=160)
    elif case == 2:
        x_0 = jnp.array([[0.0,0],[10.0,9]]) 
        v_0 = jnp.array([[1,1],[-1.0,-1]])
        x_new, v_new = solve_for((x_0,v_0), f_exponential_repulsion_par, exp_right_perp, .1, 100)

        # x_slv, v_slv = solve((x_0, v_0), lambda *args: jnp.array([0,0]), lambda *args: jnp.array([0,0]), lambda x, v: f_exponential_repulsion(x, params), 0.1, 10)

        # name = "TESTER_SOLVE.npz"
        name = "TESTER.npz"
        np.savez(
            name,
            # x=x_slv,
            # v=v_slv,
            x=x_new,
            v=v_new,
            color = ["tab:blue", "tab:red"]
        )
        render_animation_gif(name, dt=1)
    elif case ==3:
        overlay_npz("s11.npz", "s11_init_run.npz", "combined.npz")
        render_animation_gif("combined.npz", dt=0.1, enter_t=0, tail_len=2)    

    