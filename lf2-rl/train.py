import os

from env import LF2Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 训练总步数，可用环境变量覆盖（冒烟测试：LF2_TIMESTEPS=2048 python train.py）
TOTAL_TIMESTEPS = int(os.environ.get("LF2_TIMESTEPS", "200000"))
# 页面服务端口：只用于打印提示，本脚本本身不监听它（训练用的是 ws 8765，写死在 env.py 里）
HTTP_PORT = os.environ.get("LF2_PORT", "8000")
# 每多少步存一个 checkpoint（可用 LF2_CKPT_FREQ 覆盖）。40fps 下 20000 步≈8 分钟：
# 中途 Ctrl+C 最多丢这么多，不加的话整场训练白跑
CKPT_FREQ = int(os.environ.get("LF2_CKPT_FREQ", "20000"))

print("初始化环境...")
env = LF2Env()

print("请先启动静态服务和浏览器（两个终端，端口要一致）：")
print(f"  python serve.py --port {HTTP_PORT}")
print(f"  python launch_browser.py --port {HTTP_PORT}")
print("或者手动打开页面，地址为：")
print(f"  http://localhost:{HTTP_PORT}/LF2-RL/game/game.html?debug&headless")
print("（8000 被占用时：set LF2_PORT=8010 再跑本脚本；ws 桥固定 8765，不用管）")
print("等待连接...")

model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,        # 原来 512，调大让每次更新见到更多样本
    batch_size=64,
    learning_rate=1e-4,
    n_epochs=5,
    gamma=0.99,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/",
)

print("开始训练！浏览器最小化即可，训练在后台继续")
# 定期把权重存进 checkpoints/，这样中途 Ctrl+C 也不会整场白跑
checkpoint = CheckpointCallback(
    save_freq=CKPT_FREQ,
    save_path="checkpoints",
    name_prefix="ppo",
)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint)   # 原来 100k，至少跑 500k 才能看到学习效果
model.save("lf2_ppo")
print("训练完成！模型已保存为 lf2_ppo.zip")
