"""Generate reproducible hourly synthetic data; contains no external dataset dependency."""
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / "series.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["date", "target"])
    for i in range(360):
        writer.writerow([(datetime(2025, 1, 1)+timedelta(hours=i)).isoformat(),
                         20 + 0.01*i + 3*math.sin(i*math.tau/24)+0.2*math.cos(i*math.tau/7)])
print(root / "series.csv")
