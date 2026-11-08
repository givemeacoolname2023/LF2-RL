"""
launch_browser.py —— 用一个不会被浏览器“冻结”的方式打开 F.LF 游戏页面。

背景：
    游戏主循环是浏览器的 setInterval（LF/network.js），Chrome/Edge 会把
    后台/最小化/被遮挡标签页的定时器降频甚至冻结，导致训练方收不到状态、
    表现为 `[结果] 🔄 帧超时` 或 reset 卡死。

这个脚本做两件事：
    1. 默认用 `--headless=new` 启动：没有窗口，就不存在“看不见”这回事；
    2. 同时带上禁用后台节流的启动参数，即使有窗口也不会被降频（--visible 模式）。

用法（在 lf2-rl 目录下，两个终端）：
    python serve.py --port 8010                   # 终端 A：静态服务（禁用缓存）
    python launch_browser.py --port 8010          # 终端 B：无头，后台训练
    python launch_browser.py --visible            # 有窗口，但禁用节流
    python launch_browser.py --devtools-port 0    # 让 Chrome 自选调试端口

依赖：本机装有 Chrome 或 Edge。
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

# 常见的安装位置（按顺序尝试）
BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"找不到浏览器：{explicit}")
        return explicit

    env_browser = os.environ.get("LF2_BROWSER")
    if env_browser and os.path.exists(env_browser):
        return env_browser

    for path in BROWSER_CANDIDATES:
        if os.path.exists(path):
            return path

    for name in ("chrome", "chrome.exe", "msedge", "msedge.exe"):
        found = shutil.which(name)
        if found:
            return found

    raise SystemExit(
        "找不到 Chrome/Edge。请安装其中之一，或用 --browser 指定可执行文件路径，"
        "或设置环境变量 LF2_BROWSER。"
    )


def url_alive(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def kill_tree(proc):
    """结束 Chrome 及其子进程（Windows 下 terminate 只杀主进程）。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    parser = argparse.ArgumentParser(
        description="启动 F.LF 游戏页面（默认无头模式，适合无人值守训练）"
    )
    parser.add_argument("--browser", default=None, help="浏览器可执行文件路径")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 服务地址")
    parser.add_argument("--port", type=int, default=8000, help="HTTP 服务端口")
    parser.add_argument(
        "--path",
        default="/LF2-RL/game/game.html?debug&headless",
        help="游戏页面路径（相对于 HTTP 服务的根目录）",
    )
    parser.add_argument(
        "--diff",
        choices=("easy", "normal", "difficult", "0", "1", "2"),
        default=None,
        help="对手难度，可写 easy / normal / difficult 或 0 / 1 / 2，"
             "写进 URL 的 diff=0/1/2（不给就用页面默认的 easy）",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="显示窗口（仍会禁用后台节流），默认是无头",
    )
    parser.add_argument(
        "--devtools-port",
        type=int,
        default=9222,
        help="远程调试端口；设为 0 让浏览器自选（默认 9222）",
    )
    parser.add_argument(
        "--profile",
        default=os.path.join(tempfile.gettempdir(), "lf2_rl_chrome"),
        help="独立的浏览器用户目录，避免和日常使用的 Chrome 冲突",
    )
    args = parser.parse_args()

    # 难度：把 --path 里可能已经写过的 diff=N 换掉（而不是简单删掉再拼，
    # 那样 `?diff=2&debug` 会变成 `&debug?diff=0`，把 debug 也弄丢）
    if args.diff:
        # 名字和 0/1/2 两种写法都接受（这个脚本不 import env.py，
        # 免得为了一个参数去加载 gymnasium/websockets，所以就地一张小表）
        level = {"easy": 0, "normal": 1, "difficult": 2,
                 "0": 0, "1": 1, "2": 2}[args.diff]
        base, _, query = args.path.partition("?")
        kept = [p for p in query.split("&") if p and not re.match(r"diff=\d+$", p)]
        args.path = base + "?" + "&".join(kept + [f"diff={level}"])

    url = f"http://{args.host}:{args.port}{args.path}"
    browser = find_browser(args.browser)

    if not url_alive(url):
        print(f"[警告] 无法访问 {url}")
        print("       请先在本目录启动静态服务（它默认禁用浏览器缓存）：")
        print(f"       python serve.py --port {args.port}")
        print("       （端口被占用时换一个 --port，例如 8010）")
        print("       仍然继续启动浏览器，页面可能显示 404。\n")

    flags = [
        f"--user-data-dir={args.profile}",
        "--no-first-run",
        "--no-default-browser-check",
        # 关键：禁止后台标签页的定时器降频 / 冻结
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-features=IntensiveWakeUpThrottling,CalculateNativeWinOcclusion",
    ]

    if args.devtools_port != 0:
        flags.append(f"--remote-debugging-port={args.devtools_port}")

    if args.visible:
        flags.append("--new-window")
    else:
        flags.append("--headless=new")

    flags.append(url)

    print(f"[浏览器] {browser}")
    print(f"[模式]   {'有窗口（禁用节流）' if args.visible else '无头（无窗口）'}")
    print(f"[页面]   {url}")
    if args.devtools_port:
        print(f"[调试]   http://{args.host}:{args.devtools_port}  可查看页面和 Console")
    print("[提示]   按 Ctrl+C 结束浏览器\n")

    proc = subprocess.Popen([browser] + flags)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[停止] 正在关闭浏览器...")
    finally:
        kill_tree(proc)


if __name__ == "__main__":
    sys.exit(main())
