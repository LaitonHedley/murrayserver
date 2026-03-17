# util.py
import matplotlib.pyplot as plt

def plot_trajectory(env, trajectories, episode=0, testing=False):
    """
    trajectories: list of step dict lists, each step dict must have
      'ball_x', 'ball_y', 'p1_pos'
    Plots ball path(s) and the agent paddle track for a single episode.
    """
    d = env.dim
    fig, ax = plt.subplots(figsize=(8, 5))

    # Playfield bounds
    ax.set_xlim(d.frame_left, d.frame_right)
    ax.set_ylim(d.H, 0)              # invert y to match screen coords (top=0)
    ax.axhline(d.paddle_y, ls="--", lw=1, alpha=0.5, label="paddle line")

    # Plot each trajectory
    for i, traj in enumerate(trajectories):
        if not traj:
            continue
        xs = [st["ball_x"] for st in traj]
        ys = [st["ball_y"] for st in traj]
        ax.plot(xs, ys, label=f"ball path {i}")

        # Paddle center track (optional visual)
        pc = [st["p1_pos"] + d.paddle_w/2 for st in traj]
        py = [d.paddle_y] * len(traj)
        ax.plot(pc, py, alpha=0.7, lw=2, label=f"paddle {i}")

    ax.set_title(f"Episode {episode} trajectory")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if testing:
        return fig, ax
    plt.show()
