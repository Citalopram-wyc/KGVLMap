"""Read-only loopback dashboard; observes a running training job without restarting it."""
import argparse
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmap.training_history import epoch_history, validation_summary


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


def snapshot(run):
    current = read_json(run/'status.json')
    metadata = read_json(run/'run_config.json')
    launcher = read_json(run/'launcher.json')
    config = metadata.get('config', {})
    active = None
    try:
        process = psutil.Process(int(launcher['pid']))
        start = datetime.datetime.fromisoformat(launcher['start_time_utc'].replace('Z','+00:00')).timestamp()
        active = process.is_running() and abs(process.create_time()-start) < 1
    except psutil.NoSuchProcess:
        active = False
    except (psutil.AccessDenied, KeyError, ValueError):
        pass
    history = epoch_history(run)
    summary = validation_summary(history)
    phase = current.get('state', 'initializing')
    if phase == 'validating':
        phase = 'awaiting_validation'
    if active is False and phase not in ('failed','complete'):
        phase = 'stopped'
    elif phase == 'training' and current.get('update_in_epoch',0) >= current.get('updates_per_epoch',float('inf')):
        latest = summary['latest_validation']
        phase = 'saving' if latest and latest['epoch'] == current.get('epoch') else 'awaiting_validation'
    try:
        updated = datetime.datetime.fromtimestamp((run/'status.json').stat().st_mtime).isoformat(timespec='seconds')
    except OSError:
        updated = None
    return safe_json({'run':run.name,'run_directory':str(run),'phase':phase,'process_alive':active,
                      'updated_at':updated,'refreshed_at':datetime.datetime.now().isoformat(timespec='seconds'),
                      'current':current,'history':history,**summary,
                      'config':{k:config.get(k) for k in ('epochs','micro_batch','accumulation_steps')},
                      'train_tiles':metadata.get('train_tiles'),'val_tiles':metadata.get('val_tiles'),
                      'smoke_test':metadata.get('smoke_test',False)})


def publish(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for _ in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(.1)
    raise PermissionError(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    if not (run/'run_config.json').exists():
        raise FileNotFoundError(run/'run_config.json')
    html = ROOT/'tools'/'training_dashboard.html'
    lock = threading.Lock()
    state = snapshot(run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split('?',1)[0]
            if route == '/':
                body, content_type = html.read_bytes(), 'text/html; charset=utf-8'
            elif route == '/api/status':
                with lock:
                    body = json.dumps(state,ensure_ascii=False,allow_nan=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self,*args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1',args.port),Handler)

    def refresh():
        nonlocal state
        while True:
            try:
                updated = snapshot(run)
                publish(run/'training_overview.json',updated)
                with lock:
                    state = updated
            except Exception as exc:
                print(f'Dashboard refresh: {exc!r}',flush=True)
            time.sleep(5)

    threading.Thread(target=refresh,daemon=True).start()
    publish(run/'dashboard.json',{'pid':psutil.Process().pid,'started_at':datetime.datetime.now().isoformat(),
                                 'url':f'http://127.0.0.1:{args.port}','run_directory':str(run)})
    print(f'Training dashboard: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
