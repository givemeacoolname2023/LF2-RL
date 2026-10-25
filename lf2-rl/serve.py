"""
serve.py —— 给 F.LF 用的静态服务器，和 `python -m http.server` 唯一的区别是
回包带 `Cache-Control: no-store`，浏览器不会缓存 LF/*.js 和素材。

为什么需要它：
    `python -m http.server` 不发送任何缓存头，Chrome 会用“启发式缓存”
    把 LF/match.js 之类留很久。改了 JS 之后刷新页面，实际跑的仍是旧代码，
    排查时会误以为“改了没用”。用这个服务器就不存在这种情况。

用法（随便在哪个目录）：
    python serve.py                 # 默认服务 group_project 目录，端口 8000
    python serve.py --port 8010     # 8000 被占用时换端口
    python serve.py --dir D:\\path  # 指定站点根目录（该目录下要有 LF2-RL 和 LF2_19）

站点根目录默认是本文件的上一级上一级（即 LF2-RL 和 LF2_19 所在的 group_project）。
"""

import argparse
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# lf2-rl/serve.py → lf2-rl → LF2-RL → group_project
DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, max-age=0')
        self.send_header('Pragma', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        # 素材请求几百条，全打出来会淹没训练日志；只保留 4xx/5xx（404 排查用）
        if len(args) >= 2 and str(args[1])[:1] in ('4', '5'):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="F.LF 静态服务器（禁用缓存）")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--dir', default=DEFAULT_ROOT, help='站点根目录')
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f'[错误] 目录不存在：{root}')
        return 1
    if not os.path.isdir(os.path.join(root, 'LF2-RL')):
        print(f'[提示] {root} 下没有 LF2-RL 目录，页面路径可能不对。')
        print('       站点根目录应该是 LF2-RL 和 LF2_19 的上一级（group_project）。')

    os.chdir(root)
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), NoCacheHandler)
    except OSError as e:
        print(f'[错误] 端口 {args.port} 无法使用：{e}')
        print(f'       换一个端口，例如：python serve.py --port 8010')
        return 1

    print(f'[目录] {root}')
    print(f'[地址] http://{args.host}:{args.port}/LF2-RL/game/game.html?debug&headless')
    print('[提示] 按 Ctrl+C 停止')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n[停止] 服务器已关闭')
    finally:
        httpd.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
