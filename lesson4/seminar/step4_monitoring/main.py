from src.config import load_cfg
from src.monitor import health, predict
import numpy as np
import time
from colorama import Fore, Style, init
from datetime import datetime, timezone
from pythonjsonlogger import jsonlogger
import os
import json
import logging


init(autoreset=True)

os.makedirs('mlops2025/lesson4/seminar/step4_monitoring/logs', exist_ok=True)

logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO)
logfile_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
json_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
logfile_handler.setFormatter(json_formatter)
logger.addHandler(logfile_handler)

metrics_logger = logging.getLogger('metrics_logger')
metrics_logger.setLevel(logging.INFO)
metrics_handler = logging.FileHandler('logs/metrics.jsonl', encoding='utf-8')
metrics_handler.setFormatter(logging.Formatter('%(message)s'))
metrics_logger.addHandler(metrics_handler)

def log_metric_jsonl(metrics):
    metrics['timestamp'] = datetime.now(timezone.utc).isoformat()
    metrics_json = json.dumps(metrics, ensure_ascii=False)
    metrics_logger.info(metrics_json)

def p95(arr):
    return np.percentile(arr, 95)

def color_code(thresholds, metric_name, value):
    warn = thresholds[metric_name]['warning']
    crit = thresholds[metric_name]['critical']
    if value >= warn:
        return Fore.YELLOW
    elif value >= crit:
        return Fore.RED
    else:
        return Fore.GREEN

def main():
    latencies = []
    total_requests = 0
    error_count = 0
    consecutive_failures = 0

    cfg = load_cfg('src/config.py')
    url = cfg['service'].get('base_url')
    timeout = cfg['monitoring'].get('request_timeout_seconds')
    samples = int(cfg['monitoring'].get('samples_per_check', 1))

    thresholds = cfg['thresholds']

    for _ in range(samples):
        h = health(url, timeout=timeout)
        print('HEALTH:', h)

        start_time = time.perf_counter()
        p = predict(url, 'img.jpg', timeout=timeout)
        total_requests += 1
        if isinstance(p, dict) and p.get('success'):
            consecutive_failures = 0
        else:
            error_count += 1
            consecutive_failures += 1

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000
        latencies.append(latency)
        error_rate = (error_count / total_requests) * 100 if total_requests != 0 else 0
        print('PREDICT:', p)
        print()

        color_time = color_code(thresholds, 'response_time_ms', p['result']['timing']['total_ms'])
        print(f"{color_time}response_time_ms: {p['result']['timing']['total_ms']:.2f}{Style.RESET_ALL}")

        latency_95 = color_code(thresholds, 'p95_latency_ms', p95(latencies))
        print(f'{latency_95}p95_latency_ms: {p95(latencies):.2f}{Style.RESET_ALL}')

        error_color = color_code(thresholds, 'error_rate_percent', error_rate)        
        print(f'{error_color}error_rate_percent: {error_rate}{Style.RESET_ALL}')

        consec_color = color_code(thresholds, 'consecutive_failures', consecutive_failures) 
        print(f'{consec_color}consecutive_failures: {consecutive_failures}{Style.RESET_ALL}')
    
        logger.info({"event": "predict_result", "predict_response": p})

        metrics_data = {
            "response_time_ms": p['result']['timing']['total_ms'],
            "p95_latency_ms": p95(latencies),
            "error_rate_percent": error_rate,
            "consecutive_failures": consecutive_failures,
            "health_status": h
        }
        log_metric_jsonl(metrics_data)

if __name__ == "__main__":
    main()