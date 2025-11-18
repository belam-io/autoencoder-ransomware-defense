import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

NUM_ROWS = 500_000
np.random.seed(42)
random.seed(42)

# ----------------------------------------
# 1. Helper value lists
# ----------------------------------------
locations = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Kajiado", "Machakos",
    "Meru", "Nyeri", "Kisii", "Voi", "Naivasha", "Garissa", "Wajir", "Homabay"
]

tx_types = ["transfer", "cashout", "paybill", "buy_goods", "airtime"]
channels = ["ussd", "app", "pos", "agent"]

# 1,500 agents
agent_ids = [f"agent_{i}" for i in range(1, 1501)]

# 50,000 MPesa users
user_ids = [f"user_{i}" for i in range(1, 50_001)]

# ----------------------------------------
# 2. Generate timestamps
# ----------------------------------------
start_date = datetime(2024, 1, 1)
def random_timestamp():
    delta = timedelta(seconds=random.randint(0, 60*60*24*45))  # 45 days of data
    return start_date + delta

# ----------------------------------------
# 3. Dataset generation
# ----------------------------------------
rows = []

for i in range(NUM_ROWS):
    ts = random_timestamp()

    sender = random.choice(user_ids)
    receiver = random.choice(user_ids)

    amount = np.round(np.random.exponential(scale=800), 2)
    amount = np.clip(amount, 1, 70000)

    fee = np.round(amount * np.random.uniform(0.0005, 0.015), 2)

    sender_balance_before = np.round(np.random.uniform(50, 200000), 2)
    sender_balance_after = sender_balance_before - amount - fee

    receiver_balance_before = np.round(np.random.uniform(50, 200000), 2)
    receiver_balance_after = receiver_balance_before + amount

    tx_type = random.choice(tx_types)
    channel = random.choice(channels)
    agent_id = random.choice(agent_ids) if tx_type in ["cashout", "buy_goods"] else ""

    sender_loc = random.choice(locations)
    receiver_loc = random.choice(locations)

    # user "behavior signals"
    sender_age = random.randint(10, 3000)     # days
    receiver_age = random.randint(10, 3000)

    sender_freq_24h = random.randint(0, 12)
    receiver_freq_24h = random.randint(0, 12)

    rows.append([
        f"tx_{i:09}",
        ts,
        sender,
        receiver,
        amount,
        fee,
        sender_balance_before,
        sender_balance_after,
        receiver_balance_before,
        receiver_balance_after,
        tx_type,
        channel,
        agent_id,
        ts.hour,
        ts.weekday(),
        ts.day,
        ts.month,
        sender_loc,
        receiver_loc,
        sender_age,
        receiver_age,
        sender_freq_24h,
        receiver_freq_24h
    ])

# ----------------------------------------
# 4. Convert to DataFrame
# ----------------------------------------
df = pd.DataFrame(rows, columns=[
    "tx_id", "timestamp", "sender_id", "receiver_id",
    "amount", "fee",
    "sender_balance_before", "sender_balance_after",
    "receiver_balance_before", "receiver_balance_after",
    "tx_type", "channel_type", "agent_id",
    "hour", "day_of_week", "day", "month",
    "sender_location", "receiver_location",
    "sender_account_age", "receiver_account_age",
    "sender_tx_frequency_24h", "receiver_tx_frequency_24h"
])

# ----------------------------------------
# 5. Save dataset
# ----------------------------------------
df.to_csv("training_data.csv", index=False)
print("✅ Generated training_data.csv with", NUM_ROWS, "rows.")
