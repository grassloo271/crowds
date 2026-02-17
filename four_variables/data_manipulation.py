import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

# ================================================saving data=========================


def people_to_npz(crowd, name):
    """
    Takes a list of people that you want to do something and
    puts them in a npz fiile for easier reading later on
    """
    gen_arr = [(ppl.x, ppl.v, ppl.color, ppl.goal) for ppl in crowd]
    x, v, color, goal = zip(*gen_arr)
    np.savez(
        name,
        x=np.asarray(x),
        v=np.asarray(v),
        goal=np.asarray(goal),
        color=np.asarray(color),
    )


# ====================================rendering data==================================


def render_animation_gif(
    npz_file_name,
    dt: float,
    fps: int = 30,
    tail_len: int = 250,  # number of past frames to show as trails
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
    print(v_dir)
    x_parallel = jnp.sum(v_dir * x, axis=-1, keepdims=True) * v_dir
    x_perp = x - x_parallel
    return x_parallel, x_perp


def npz_to_projected(npz_file_name):
    """
    takes the npz data and turns it into the x_par, x_perp, v_par, v_perp
    """
    data = np.load(npz_file_name)
    x = jnp.array(data["x"])
    v = jnp.array(data["v"])

    x_diff = x[:, None, :] - x[:, :, None]
    v_diff = v[:, None, :] - v[:, :, None]

    v_par, v_perp = project(v_diff, v)
    x_par, x_perp = project(x_diff, x)
    return x_par, x_perp, v_par, v_perp


def projected_to_absolute(f_par, f_perp, v):
    """
    turns the projected forces into absolutes to easier deal with in the euler.
    """
    eps = 1e-6
    v_dir = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)
    rot = jnp.array([[0, -1], [1, 0]])
    v_perp = (rot @ v_dir.T).T
    f_pa = f_par[:, :, None] * v_dir[None, :, :]
    f_per = f_perp[:, :, None] * v_perp[None, :, :]
    return f_pa, f_per


# =======================================end of functions===============================


if __name__ == "__main__":
    # x,y= project(jnp.array([[[2,0],[1,0]],[[0,1],[0,3]]]), jnp.array([[0,1],[1,0]]))
    f1 = jnp.array([[0, 2], [3, 0]])
    f2 = jnp.array([[1, 1], [1, 1]])
    z = jnp.array([[0, 1], [3, 4]])
    a, b = projected_to_absolute(f1, f2, z)
    print(f"{a=}")
    print(f"{b=}")
