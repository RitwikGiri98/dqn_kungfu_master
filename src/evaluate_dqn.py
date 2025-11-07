# src/evaluate_dqn.py
import argparse, os, cv2, numpy as np, torch
from src.wrappers import make_kungfu_env
from src.q_network import DQN
from src.utils import get_device

def parse_args():
    ap = argparse.ArgumentParser("Evaluate DQN on ALE/KungFuMaster-v5")
    ap.add_argument("--weights", type=str, default="outputs/trained_model.pth")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--record", type=str, default="", help="Path to .mp4 to record gameplay (RGB)")
    return ap.parse_args()

def main():
    args = parse_args()
    # env for viewing/recording should render RGB
    env = make_kungfu_env(render_mode="rgb_array")
    obs, _ = env.reset()
    n_actions, in_ch = env.action_space.n, obs.shape[0]
    device = get_device()

    net = DQN(in_ch, n_actions).to(device)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    if not args.weights or not os.path.exists(args.weights):
        print("⚠️ No weights provided — running with random policy (baseline).")
        net = None
    net.eval()

    writer = None
    if args.record:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        # infer frame size from env render
        frame = env.render()
        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 30, (w, h))

    scores = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        total = 0.0
        for t in range(args.steps):
            with torch.no_grad():
                s = torch.from_numpy(obs).unsqueeze(0).float().to(device) / 255.0
                if net is None:
                    a = env.action_space.sample()  # random baseline
                else:
                    a = int(net(s).argmax(1).item())

            obs, r, term, trunc, _ = env.step(a)
            total += r

            if args.render:
                frm = env.render()
                cv2.imshow("KungFuMaster", frm[:, :, ::-1])  # RGB->BGR for imshow
                cv2.waitKey(1)
            if writer is not None:
                frm = env.render()
                writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))

            if term or trunc:
                break
        print(f"Episode {ep+1}: reward={total}")
        scores.append(total)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Average reward over eval:", np.mean(scores))

if __name__ == "__main__":
    main()
