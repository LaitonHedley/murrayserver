from speedyibl import Agent
from ts_env_min import TeamSpiritEnvMin, ACTIONS  # ACTIONS = ("LEFT","STAY","RIGHT")

# 1) env + agent
env = TeamSpiritEnvMin(seed=42, block_type="col")
agent = Agent(default_utility=15, noise=0.25, decay=0.5)


def run_ibl_episode(agent: Agent, env: TeamSpiritEnvMin, max_steps: int = 3000):
    """
    One full episode with proper EDF wiring:
      - At each step: choose, env.step, respond(r_step)
      - At the end: equal_delay_feedback(total_return, episode_history)
    Returns: final_r, total_return, episode_history, trajectory
    """
    s = env.reset()
    episode_history = []   # list of (state, action, r_step, time_index)
    trajectory = []        # optional: track a few raw vars for plotting/debug
    rewards = []
    final_r = 0.0

    for t in range(max_steps):
        # options use your actual (state, action) pairs (strings), not hashes or ints
        options = [(s, a) for a in ACTIONS]
        s_chosen, action = agent.choose(options)
        

        # step the env
        s_next, r_step, done, _ = env.step(action)

        # IMPORTANT: log the *same* r_step you used in respond() into episode_history
        agent.respond(r_step)
        rewards.append(r_step)
        episode_history.append((s_chosen, action, r_step, t))
        # trajectory.append({"t": t, "state": s_chosen, "action": action})
        # # after: s_next, r_step, done, _ = env.step(action)
        rs = env.raw_state()  # has ball_x, ball_y, p1_pos, etc.
        trajectory.append({
            "t": t,
            "state": s_chosen,
            "action": action,
            "ball_x": rs["ball_x"],
            "ball_y": rs["ball_y"],
            "p1_pos": rs["p1_pos"],
        })
        
        # consistent, defined logging (avoid undefined s/a/act)
        print(f"CompActivation(t={t+1}, option=({s_chosen!r}, {action!r}))")
                

        final_r = r_step
        s = s_next

        if done:
            break

    total_return = float(sum(rewards))
    # Retroactively assign the episode return to all the steps we logged above
    agent.equal_delay_feedback(total_return, episode_history)
    return final_r, total_return, episode_history, trajectory


final_r, total_return, episode_history, trajectory = run_ibl_episode(agent, env)
print(f"steps={len(episode_history)}, terminal_r={final_r:+.1f}, return={total_return:+.1f}")



# # 2) get a state
# s = env.reset()
# print("ACTIONS:", ACTIONS)
# print("state:", s)

# # 3) show internal activation per action at the next time step
# t = getattr(agent, "t", 0)  # current internal time if present, else 0
# for a in ACTIONS:
#     act = agent.CompActivation(t + 1, (s, a))
#     print(f"CompActivation(t={t+1}, option=({s!r}, {a!r})) -> {act}")

# # 4) make one actual choice the *normal* way
# options = [(s, a) for a in ACTIONS]
# _, chosen = agent.choose(options)
# print("chosen action:", chosen)

# # 5) log a placeholder outcome so the agent stores an instance (needed before delayed credit)
# agent.respond(0.0)

# 6) peek at memory size
# print("instances stored:", len(agent.instances()))
