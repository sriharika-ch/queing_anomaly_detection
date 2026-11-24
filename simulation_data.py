
import csv
import math
import random
import time
from collections import deque
from datetime import datetime, timedelta

# ============================================================
# 1. SIMULATION CONFIGURATION
# ============================================================

SIMULATION_DURATION = 2400       # 40 minutes
DT = 0.1                         # Simulation step: 0.1 seconds
RECORD_INTERVAL = 1              # Write one CSV row every second

QUEUE_CAPACITY = 25              # Fixed throughout the simulation
OUTPUT_FILE = "banking_queue_dataset.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TRANSACTION_TYPES = {
    "UPI": {
        "weight": 0.30,
        "mean_service": 1.5
    },
    "CARD_PAYMENT": {
        "weight": 0.22,
        "mean_service": 2.0
    },
    "ATM": {
        "weight": 0.12,
        "mean_service": 2.5
    },
    "IMPS": {
        "weight": 0.14,
        "mean_service": 2.2
    },
    "NEFT": {
        "weight": 0.12,
        "mean_service": 3.0
    },
    "NET_BANKING": {
        "weight": 0.10,
        "mean_service": 2.4
    }
}

TRANSACTION_NAMES = list(TRANSACTION_TYPES.keys())
TRANSACTION_WEIGHTS = [
    TRANSACTION_TYPES[name]["weight"]
    for name in TRANSACTION_NAMES
]

# ============================================================
# 2. SCENARIO CONFIGURATION
# ============================================================

SCENARIOS = [
    {
        "name": "NORMAL",
        "start": 0,
        "end": 360,
        "arrival_rate": 0.32,
        "service_multiplier": 1.0,
        "patience": 90.0,
        "is_anomaly": 0
    },
    {
        "name": "ARRIVAL_SURGE",
        "start": 360,
        "end": 720,
        "arrival_rate": 1.05,
        "service_multiplier": 1.0,
        "patience": 90.0,
        "is_anomaly": 1
    },
    {
        "name": "NORMAL",
        "start": 720,
        "end": 1080,
        "arrival_rate": 0.32,
        "service_multiplier": 1.0,
        "patience": 90.0,
        "is_anomaly": 0
    },
    {
        "name": "SERVICE_SLOWDOWN",
        "start": 1080,
        "end": 1440,
        "arrival_rate": 0.32,
        "service_multiplier": 2.8,
        "patience": 90.0,
        "is_anomaly": 1
    },
    {
        "name": "LONG_SERVICE_BLOCKAGE",
        "start": 1440,
        "end": 1800,
        "arrival_rate": 0.32,
        "service_multiplier": 8.0,
        "patience": 90.0,
        "is_anomaly": 1
    },
    {
        "name": "BUFFER_OVERFLOW",
        "start": 1800,
        "end": 2160,
        "arrival_rate": 1.25,
        "service_multiplier": 1.0,
        "patience": 90.0,
        "is_anomaly": 1
    },
    {
        "name": "CUSTOMER_ABANDONMENT",
        "start": 2160,
        "end": 2280,
        "arrival_rate": 0.72,
        "service_multiplier": 1.0,
        "patience": 7.0,
        "is_anomaly": 1
    },
    {
        "name": "SYSTEM_RECOVERY",
        "start": 2280,
        "end": 2400,
        "arrival_rate": 0.12,
        "service_multiplier": 1.0,
        "patience": 90.0,
        "is_anomaly": 0
    }
]

def get_scenario(elapsed_time):
    for scenario in SCENARIOS:
        if scenario["start"] <= elapsed_time < scenario["end"]:
            return scenario

    return SCENARIOS[-1]

# ============================================================
# 3. POISSON ARRIVAL GENERATOR
# ============================================================

def poisson_random(lam):
    if lam <= 0:
        return 0

    limit = math.exp(-lam)
    product = 1.0
    count = 0

    while product > limit:
        count += 1
        product *= random.random()

    return count - 1

# ============================================================
# 4. SIMULATION STATE
# ============================================================

state = {
    "queue": deque(),
    "server": None,

    "total_arrivals": 0,
    "total_served": 0,
    "total_dropped": 0,
    "total_abandoned": 0
}

# ============================================================
# 5. PER-SECOND METRICS
# ============================================================

def create_interval_metrics():

    metrics = {
        "arrivals": 0,
        "served": 0,
        "dropped": 0,
        "abandoned": 0,

        "busy_seconds": 0.0,

        "completed_waits": [],
        "completed_services": [],

        "started_waits": [],

        "queue_sum": 0.0,
        "occupancy_sum": 0.0,
        "samples": 0,

        "max_queue": 0,
        "max_occupancy": 0
    }

    for name in TRANSACTION_NAMES:
        metrics[f"arrivals_{name}"] = 0
        metrics[f"served_{name}"] = 0
        metrics[f"dropped_{name}"] = 0

    return metrics

# ============================================================
# 6. START SERVICE
# ============================================================

def start_next_transaction(current_time, scenario, metrics):

    if state["server"] is not None:
        return

    if not state["queue"]:
        return

    customer = state["queue"].popleft()

    nominal_mean = customer["nominal_service_seconds"]

    effective_mean = (
        nominal_mean * scenario["service_multiplier"]
    )

    # Exponential service-time distribution.
    # Expected service time = effective_mean.
    service_time = random.expovariate(
        1.0 / effective_mean
    )

    waiting_time = max(
        0.0,
        current_time - customer["arrival_time"]
    )

    metrics["started_waits"].append(waiting_time)

    state["server"] = {
        "transaction": customer,
        "remaining_service": service_time,
        "original_service_time": service_time,
        "waiting_time": waiting_time
    }

# ============================================================
# 7. ABANDONMENT HANDLING
# ============================================================

def process_abandonment(current_time, metrics):

    remaining_queue = deque()

    while state["queue"]:

        customer = state["queue"].popleft()

        waiting_time = (
            current_time - customer["arrival_time"]
        )

        if waiting_time >= customer["patience_seconds"]:

            state["total_abandoned"] += 1
            metrics["abandoned"] += 1

        else:
            remaining_queue.append(customer)

    state["queue"] = remaining_queue

# ============================================================
# 8. ONE SIMULATION STEP
# ============================================================

def simulation_step(step_number, metrics):

    current_time = step_number * DT

    scenario = get_scenario(current_time)

    # --------------------------------------------------------
    # A. Process the transaction currently being served
    # --------------------------------------------------------

    if state["server"] is not None:

        server = state["server"]

        remaining_before = server["remaining_service"]

        busy_time = min(DT, remaining_before)

        metrics["busy_seconds"] += busy_time

        server["remaining_service"] -= DT

        if server["remaining_service"] <= 0:

            transaction = server["transaction"]

            state["total_served"] += 1
            metrics["served"] += 1

            metrics["completed_waits"].append(
                server["waiting_time"]
            )

            metrics["completed_services"].append(
                server["original_service_time"]
            )

            transaction_type = transaction["transaction_type"]

            metrics[f"served_{transaction_type}"] += 1

            state["server"] = None

    # --------------------------------------------------------
    # B. Process customers who have abandoned the queue
    # --------------------------------------------------------

    end_of_step = current_time + DT

    process_abandonment(end_of_step, metrics)

    # --------------------------------------------------------
    # C. Generate Poisson arrivals
    # --------------------------------------------------------

    arrival_count = poisson_random(
        scenario["arrival_rate"] * DT
    )

    for _ in range(arrival_count):

        transaction_type = random.choices(
            TRANSACTION_NAMES,
            weights=TRANSACTION_WEIGHTS,
            k=1
        )[0]

        transaction = {
            "transaction_type": transaction_type,
            "arrival_time": end_of_step,
            "nominal_service_seconds":
                TRANSACTION_TYPES[transaction_type]["mean_service"],
            "patience_seconds": scenario["patience"]
        }

        state["total_arrivals"] += 1
        metrics["arrivals"] += 1

        metrics[f"arrivals_{transaction_type}"] += 1

        # Capacity includes the active server and waiting queue.
        current_occupancy = (
            len(state["queue"])
            + (1 if state["server"] is not None else 0)
        )

        if current_occupancy < QUEUE_CAPACITY:

            state["queue"].append(transaction)

        else:

            state["total_dropped"] += 1
            metrics["dropped"] += 1

            metrics[f"dropped_{transaction_type}"] += 1

    # --------------------------------------------------------
    # D. Start the next transaction if the server is idle
    # --------------------------------------------------------

    start_next_transaction(
        end_of_step,
        scenario,
        metrics
    )

    # --------------------------------------------------------
    # E. Record queue and system occupancy measurements
    # --------------------------------------------------------

    queue_length = len(state["queue"])

    occupancy = (
        queue_length
        + (1 if state["server"] is not None else 0)
    )

    metrics["queue_sum"] += queue_length
    metrics["occupancy_sum"] += occupancy
    metrics["samples"] += 1

    metrics["max_queue"] = max(
        metrics["max_queue"],
        queue_length
    )

    metrics["max_occupancy"] = max(
        metrics["max_occupancy"],
        occupancy
    )

    return scenario

# ============================================================
# 9. CREATE CSV COLUMNS
# ============================================================

fieldnames = [
    "timestamp",
    "elapsed_seconds",
    "scenario",
    "anomaly_type",
    "is_anomaly",

    "queue_capacity",
    "queue_length",
    "system_occupancy",

    "configured_arrival_rate",
    "service_time_distribution",
    "service_multiplier",

    "arrivals_this_second",
    "served_this_second",
    "dropped_this_second",
    "abandoned_this_second",

    "total_arrivals",
    "total_served",
    "total_dropped",
    "total_abandoned",

    "average_queue_length",
    "average_system_occupancy",

    "maximum_queue_length",
    "maximum_system_occupancy",

    "average_waiting_time",
    "maximum_waiting_time",
    "average_service_time",

    "server_utilization",
    "throughput_per_second",
    "drop_rate",
    "abandonment_rate"
]

for name in TRANSACTION_NAMES:
    fieldnames.extend([
        f"arrivals_{name}",
        f"served_{name}",
        f"dropped_{name}"
    ])

# ============================================================
# 10. CREATE A CSV ROW
# ============================================================

def create_csv_row(elapsed_second, start_datetime, metrics):

    # The row represents the preceding one-second interval.
    scenario = get_scenario(elapsed_second - 0.000001)

    queue_length = len(state["queue"])

    occupancy = (
        queue_length
        + (1 if state["server"] is not None else 0)
    )

    samples = max(1, metrics["samples"])

    average_queue = (
        metrics["queue_sum"] / samples
    )

    average_occupancy = (
        metrics["occupancy_sum"] / samples
    )

    average_wait = (
        sum(metrics["completed_waits"])
        / len(metrics["completed_waits"])
        if metrics["completed_waits"]
        else 0.0
    )

    maximum_wait = (
        max(metrics["started_waits"])
        if metrics["started_waits"]
        else 0.0
    )

    average_service = (
        sum(metrics["completed_services"])
        / len(metrics["completed_services"])
        if metrics["completed_services"]
        else 0.0
    )

    utilization = min(
        1.0,
        metrics["busy_seconds"] / RECORD_INTERVAL
    )

    arrivals = metrics["arrivals"]
    dropped = metrics["dropped"]
    abandoned = metrics["abandoned"]

    drop_rate = (
        dropped / arrivals
        if arrivals > 0
        else 0.0
    )

    abandonment_rate = (
        abandoned / arrivals
        if arrivals > 0
        else 0.0
    )

    timestamp = (
        start_datetime
        + timedelta(seconds=elapsed_second)
    ).isoformat(timespec="seconds")

    row = {
        "timestamp": timestamp,
        "elapsed_seconds": elapsed_second,

        "scenario": scenario["name"],

        "anomaly_type": (
            scenario["name"]
            if scenario["is_anomaly"] == 1
            else "NONE"
        ),

        "is_anomaly": scenario["is_anomaly"],

        "queue_capacity": QUEUE_CAPACITY,
        "queue_length": queue_length,
        "system_occupancy": occupancy,

        "configured_arrival_rate": scenario["arrival_rate"],
        "service_time_distribution": "EXPONENTIAL",
        "service_multiplier": scenario["service_multiplier"],

        "arrivals_this_second": arrivals,
        "served_this_second": metrics["served"],
        "dropped_this_second": dropped,
        "abandoned_this_second": abandoned,

        "total_arrivals": state["total_arrivals"],
        "total_served": state["total_served"],
        "total_dropped": state["total_dropped"],
        "total_abandoned": state["total_abandoned"],

        "average_queue_length": round(average_queue, 4),
        "average_system_occupancy": round(average_occupancy, 4),

        "maximum_queue_length": metrics["max_queue"],
        "maximum_system_occupancy": metrics["max_occupancy"],

        "average_waiting_time": round(average_wait, 4),
        "maximum_waiting_time": round(maximum_wait, 4),
        "average_service_time": round(average_service, 4),

        "server_utilization": round(utilization, 4),
        "throughput_per_second": metrics["served"] / RECORD_INTERVAL,

        "drop_rate": round(drop_rate, 4),
        "abandonment_rate": round(abandonment_rate, 4)
    }

    for name in TRANSACTION_NAMES:
        row[f"arrivals_{name}"] = metrics[f"arrivals_{name}"]
        row[f"served_{name}"] = metrics[f"served_{name}"]
        row[f"dropped_{name}"] = metrics[f"dropped_{name}"]

    return row

# ============================================================
# 11. MAIN SIMULATION
# ============================================================

def run_simulation():

    start_datetime = datetime.now()
    start_monotonic = time.monotonic()

    total_steps = int(SIMULATION_DURATION / DT)

    rows_written = 0

    print("\nBanking transaction queue simulation started.")
    print("Duration: 40 minutes")
    print("Queue capacity: 25")
    print("Service-time distribution: Exponential")
    print("CSV file:", OUTPUT_FILE)
    print("\nSimulation progress:\n")

    with open(
        OUTPUT_FILE,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        csv_file.flush()

        metrics = create_interval_metrics()

        for step in range(total_steps):

            simulation_step(step, metrics)

            # Write one row after every simulated second.
            if (step + 1) % int(RECORD_INTERVAL / DT) == 0:

                elapsed_second = int(
                    (step + 1) * DT
                )

                row = create_csv_row(
                    elapsed_second,
                    start_datetime,
                    metrics
                )

                writer.writerow(row)
                csv_file.flush()

                rows_written += 1

                print(
                    f"Elapsed: {elapsed_second // 60:02d}:"
                    f"{elapsed_second % 60:02d} | "
                    f"Scenario: {row['scenario']} | "
                    f"Queue: {row['queue_length']} | "
                    f"Occupancy: {row['system_occupancy']} | "
                    f"Arrivals: {row['arrivals_this_second']} | "
                    f"Served: {row['served_this_second']} | "
                    f"Dropped: {row['dropped_this_second']} | "
                    f"Abandoned: {row['abandoned_this_second']}"
                )

                metrics = create_interval_metrics()

            # Maintain approximately real-time execution.
            next_deadline = (
                start_monotonic + (step + 1) * DT
            )

            remaining_time = (
                next_deadline - time.monotonic()
            )

            if remaining_time > 0:
                time.sleep(remaining_time)

    # ========================================================
    # 12. FINAL VALIDATION
    # ========================================================

    final_occupancy = (
        len(state["queue"])
        + (1 if state["server"] is not None else 0)
    )

    accounted_transactions = (
        state["total_served"]
        + state["total_dropped"]
        + state["total_abandoned"]
        + final_occupancy
    )

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)

    print("CSV file:", OUTPUT_FILE)
    print("Rows written:", rows_written)
    print("Expected rows:", SIMULATION_DURATION)
    print("Fixed queue capacity:", QUEUE_CAPACITY)

    print("\nTransaction summary:")
    print("Total arrivals:", state["total_arrivals"])
    print("Total served:", state["total_served"])
    print("Total dropped:", state["total_dropped"])
    print("Total abandoned:", state["total_abandoned"])
    print("Transactions remaining:", final_occupancy)

    print("\nValidation:")

    print(
        "Correct row count:",
        rows_written == SIMULATION_DURATION
    )

    print(
        "Transaction accounting correct:",
        state["total_arrivals"] == accounted_transactions
    )

    print(
        "Queue capacity respected:",
        final_occupancy <= QUEUE_CAPACITY
    )

    print(
        "Fixed capacity used throughout:",
        QUEUE_CAPACITY == 25
    )

    print("\nDataset saved successfully.")

# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":
    run_simulation()