import sys
import time
from pathlib import Path

HEARTBEAT_FILE = Path("data/heartbeat.txt")
MAX_AGE_SECONDS = 120


def check():
    if not HEARTBEAT_FILE.exists():
        print("UNHEALTHY: No heartbeat file")
        sys.exit(1)

    try:
        last_beat = float(HEARTBEAT_FILE.read_text().strip())
        age = time.time() - last_beat
        if age > MAX_AGE_SECONDS:
            print(f"UNHEALTHY: Heartbeat age {age:.0f}s > {MAX_AGE_SECONDS}s")
            sys.exit(1)
        print(f"HEALTHY: Heartbeat {age:.0f}s ago")
        sys.exit(0)
    except Exception as e:
        print(f"UNHEALTHY: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check()
