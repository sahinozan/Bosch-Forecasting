"""Independent fictional demand, constructed without reading original data."""
import csv
from datetime import date, timedelta
from math import pi, sin
from pathlib import Path
import random


def generate(destination=None):
    destination = Path(destination or Path(__file__).parent / "examples")
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    with (destination / "demand.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["date", "item_id", "quantity"])
        for week in range(96):
            for item in range(1, 5):
                quantity = round(35 + 8 * item + week * 0.1 + 9 * sin(2 * pi * week / 12)
                                 + rng.uniform(-3, 3), 2)
                writer.writerow([(date(2020, 1, 6) + timedelta(weeks=week)).isoformat(),
                                 f"SYN-ITEM-{item:02}", quantity])


if __name__ == "__main__":
    generate()
