# src/train_dqn.py
import argparse, os, csv
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

# ensure ALE envs are registered even when running as a script
import ale_py  # noqa: F401

from src.wrappers import make_kungfu_env
from src.q_network import DQN
from src.replay_buffer import ReplayBuffer
from src.utils import get_device, set_seed


def eps_linear(step, start=1.0, end=0.01, decay=50_000):
    """Linear epsilon schedule."""
    if decay <= 0:
        return end
    return max(end, start - (start - end) * (step / decay))


def softmax_probs(q_tensor: torch.Tensor, tau: float) -> np.ndarray:
    """Boltzmann (softmax) over Q-values. q_tensor shape: [1, n_actions]."""
    tau = max(float(tau), 1e-6)  # avoid divide-by-zero
    prefs = q_tensor / tau
    probs = torch.softmax(prefs, dim=1)
    return probs.detach().cpu().numpy()[0]


def parse_args():
    ap = argparse.ArgumentParser(description="DQN training for ALE/KungFuMaster-v5")
    # run control
    ap.add_argument("--reward_mode", choices=["clipped", "raw"], default="clipped", help="Reward handling mode")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    # rl hyperparams
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--buffer", type=int, default=10_000)
    ap.add_argument("--target_sync", type=int, default=1000)
    # exploration
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.01)
    ap.add_argument("--eps_decay", type=int, default=50_000)
    # policy choice
    ap.add_argument("--policy", choices=["egreedy", "softmax"], default="egreedy",
                    help="Action-selection policy: epsilon-greedy or softmax(Boltzmann)")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="Temperature for softmax policy (higher = more random)")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # ---- env ----
    env = make_kungfu_env()  # 84x84 gray, stack=4, repeat=4, reward clipping
    obs, _ = env.reset()
    in_ch = obs.shape[0]
    n_actions = env.action_space.n

    # ---- networks ----
    policy_net = DQN(in_ch, n_actions).to(device)
    target_net = DQN(in_ch, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # ---- opt/loss ----
    opt = optim.Adam(policy_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    # ---- replay ----
    rb = ReplayBuffer(args.buffer, obs_shape=obs.shape)

    # ---- logging ----
    os.makedirs("outputs", exist_ok=True)
    metrics_path = "outputs/metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "epsilon", "steps"])

    # ---- train loop ----
    global_step = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0  # <-- (you asked where: initialize here, at start of each episode)

        for t in range(args.steps):
            ep_steps += 1
            eps = eps_linear(global_step, args.eps_start, args.eps_end, args.eps_decay)

            # -------- action selection --------
            if args.policy == "egreedy":
                # epsilon-greedy
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        s = torch.from_numpy(obs).unsqueeze(0).float().to(device) / 255.0
                        q = policy_net(s)
                        action = int(torch.argmax(q, dim=1).item())
            else:
                # softmax (Boltzmann) policy
                with torch.no_grad():
                    s = torch.from_numpy(obs).unsqueeze(0).float().to(device) / 255.0
                    q = policy_net(s)
                    p = softmax_probs(q, tau=args.temp)
                action = int(np.random.choice(np.arange(n_actions), p=p))
            # -------- env step --------
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if args.reward_mode == "clipped":
                reward = np.sign(reward)  # clip reward to -1, 0, or +1
            elif args.reward_mode == "raw":
                reward = reward  # raw reward
            else:
                raise ValueError(f"Invalid reward mode: {args.reward_mode}")
            done = terminated or truncated

            # store transition
            rb.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            global_step += 1

            # -------- learn --------
            if len(rb) >= args.batch:
                s, a, r, ns, d = rb.sample(args.batch)

                s_t  = torch.tensor(s, dtype=torch.float32, device=device)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                a_t  = torch.tensor(a, dtype=torch.long,   device=device)
                r_t  = torch.tensor(r, dtype=torch.float32, device=device)
                d_t  = torch.tensor(d, dtype=torch.float32, device=device)

                q_sa = policy_net(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    max_next = target_net(ns_t).max(1)[0]
                    target_q = r_t + (1.0 - d_t) * args.gamma * max_next

                loss = loss_fn(q_sa, target_q)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if global_step % args.target_sync == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # ---- episode log ----
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, ep_reward, round(eps, 6), ep_steps])

        print(f"Episode {ep+1}/{args.episodes} | reward={ep_reward:.2f} | "
              f"eps={eps:.3f} | steps={ep_steps}")

    # ---- save model ----
    torch.save(policy_net.state_dict(), "outputs/trained_model.pth")
    print("Saved outputs/trained_model.pth and outputs/metrics.csv")


if __name__ == "__main__":
    main()
