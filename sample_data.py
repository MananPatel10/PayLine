"""
Sample Bank Statement Generator for SmartLedger.

Generates realistic Indian bank statement CSVs with varied transaction
types, merchants, and spending patterns for demo/testing purposes.

Usage:
    python sample_data.py                  # Generate 3 months of data
    python sample_data.py --months 6       # Generate 6 months
    python sample_data.py --output my.csv  # Custom output filename
"""

import argparse
import csv
import random
from datetime import datetime, timedelta

random.seed(42)  # Reproducible output

SALARY = [
    ("NEFT-SALARY-TECHCORP INDIA PVT LTD", (75000, 92000), "Credit", 1),
]

FOOD_DINING = [
    ("UPI-SWIGGY-ORDER{}", (150, 650), "Debit", 8),
    ("UPI-ZOMATO-ORDER{}", (180, 750), "Debit", 6),
    ("MCDONALDS MG ROAD", (250, 550), "Debit", 2),
    ("DOMINOS PIZZA ONLINE", (300, 650), "Debit", 2),
    ("UPI-EATCLUB-LUNCH{}", (120, 320), "Debit", 4),
    ("STARBUCKS KORAMANGALA", (280, 550), "Debit", 2),
    ("TEA POST HSR LAYOUT", (50, 150), "Debit", 4),
    ("DARSHINI RESTAURANT", (80, 220), "Debit", 5),
    ("KFC INDIRANAGAR", (220, 480), "Debit", 1),
]

GROCERIES = [
    ("UPI-BIGBASKET-ORDER{}", (600, 2200), "Debit", 3),
    ("UPI-BLINKIT-QUICK{}", (150, 650), "Debit", 4),
    ("UPI-ZEPTO-DELIVERY{}", (120, 550), "Debit", 3),
    ("MORE SUPERMARKET HSR", (350, 1200), "Debit", 2),
    ("D MART WHITEFIELD", (800, 2500), "Debit", 1),
    ("RELIANCE FRESH MKT", (300, 900), "Debit", 1),
]

TRANSPORT = [
    ("UPI-UBER INDIA-RIDE{}", (120, 420), "Debit", 6),
    ("UPI-OLA CABS-RIDE{}", (100, 380), "Debit", 4),
    ("UPI-RAPIDO-BIKE{}", (40, 120), "Debit", 4),
    ("HP PETROL PUMP SARJAPUR", (500, 1800), "Debit", 2),
    ("INDIAN OIL MARATHAHALLI", (600, 1500), "Debit", 1),
    ("IRCTC ETICKET BOOKING", (450, 1800), "Debit", 1),
]

UTILITIES = [
    ("BILL/ELECTRICITY/BESCOM", (1200, 2800), "Debit", 1),
    ("BILL/BROADBAND/ACT FIBERNET", (850, 1150), "Debit", 1),
    ("JIO PREPAID RECHARGE", (299, 749), "Debit", 1),
    ("AIRTEL POSTPAID BILL", (499, 999), "Debit", 1),
    ("INDANE GAS BOOKING", (850, 950), "Debit", 1),
]

SUBSCRIPTIONS = [
    ("NETFLIX.COM SUBSCRIPTION", (499, 649), "Debit", 1),
    ("SPOTIFY INDIA PREMIUM", (119, 179), "Debit", 1),
    ("AMAZON PRIME MEMBERSHIP", (299, 1499), "Debit", 1),
    ("HOTSTAR VIP RENEWAL", (299, 499), "Debit", 1),
    ("YOUTUBE PREMIUM INDIA", (129, 189), "Debit", 1),
]

EMI_LOANS = [
    ("EMI/HDFC/PERSONAL-LOAN-8821", (12000, 16000), "Debit", 1),
]

RENT = [
    ("NEFT-RENT-MR RAMESH SHARMA", (18000, 24000), "Debit", 1),
]

HEALTHCARE = [
    ("APOLLO PHARMACY ONLINE", (180, 1200), "Debit", 1),
    ("PRACTO CONSULT DR MEHTA", (500, 900), "Debit", 1),
    ("1MG PHARMACY ORDER{}", (150, 650), "Debit", 1),
]

SHOPPING = [
    ("AMAZON.IN MARKETPLACE{}", (350, 2500), "Debit", 3),
    ("FLIPKART ONLINE{}", (300, 2200), "Debit", 2),
    ("MYNTRA FASHION ORDER{}", (450, 1800), "Debit", 1),
    ("DECATHLON SPORTS BANG", (400, 1500), "Debit", 1),
]

ENTERTAINMENT = [
    ("BOOKMYSHOW MOVIE TKT", (250, 750), "Debit", 1),
    ("PVR CINEMAS FORUM MALL", (350, 900), "Debit", 1),
]

DAILY_CATEGORIES = [
    FOOD_DINING,
    GROCERIES,
    TRANSPORT,
    SHOPPING,
    ENTERTAINMENT,
    HEALTHCARE,
]


def generate_statement(months=3, output_file="sample_statement.csv"):
    """Generate a realistic Indian bank statement CSV."""
    transactions = []
    end_date = datetime(2025, 9, 30)
    start_date = datetime(2025, 7, 1)

    # Base monthly salary & fixed costs
    base_salary = random.randint(78000, 88000)
    rent_amount = random.randint(20000, 23000)
    emi_amount = random.randint(12500, 14500)

    # Opening balance
    balance = random.randint(65000, 95000)

    current_date = start_date

    while current_date <= end_date:
        day_txns = []

        # 1st of month: Salary credit
        if current_date.day == 1:
            amt = base_salary + random.randint(-150, 300)
            balance += amt
            day_txns.append((current_date, SALARY[0][0], amt, balance, "Credit"))

        # 5th of month: Rent
        if current_date.day == 5:
            balance -= rent_amount
            day_txns.append((current_date, RENT[0][0], rent_amount, balance, "Debit"))

        # 10th of month: EMI
        if current_date.day == 10:
            balance -= emi_amount
            day_txns.append((current_date, EMI_LOANS[0][0], emi_amount, balance, "Debit"))

        # 15th of month: Utility bill
        if current_date.day == 15:
            util = random.choice(UTILITIES)
            amt = random.randint(util[1][0], util[1][1])
            balance -= amt
            day_txns.append((current_date, util[0], amt, balance, "Debit"))

        # 20th of month: Subscriptions
        if current_date.day == 20:
            sub = random.choice(SUBSCRIPTIONS)
            amt = random.randint(sub[1][0], sub[1][1])
            balance -= amt
            day_txns.append((current_date, sub[0], amt, balance, "Debit"))

        # 25th of month: Occasional ATM cash withdrawal
        if current_date.day == 25 and random.random() < 0.7:
            amt = random.choice([2000, 3000, 5000])
            balance -= amt
            day_txns.append((current_date, f"ATM/CASH WDL/SBI/KORA{random.randint(100, 999)}", amt, balance, "Debit"))

        # Daily transactions: 0 to 2 small purchases per day
        daily_count = random.choices([0, 1, 2], weights=[25, 50, 25])[0]
        for _ in range(daily_count):
            cat_list = random.choice(DAILY_CATEGORIES)
            weights = [t[3] for t in cat_list]
            template = random.choices(cat_list, weights=weights)[0]
            desc_tmpl, (lo, hi), typ, _ = template

            desc = desc_tmpl.format(random.randint(1000, 9999)) if "{}" in desc_tmpl else desc_tmpl
            amt = random.randint(lo, hi)
            balance -= amt
            day_txns.append((current_date, desc, amt, balance, "Debit"))

        transactions.extend(day_txns)
        current_date += timedelta(days=1)

    # Injected intentional anomalies for AI anomaly detection demo:
    # 1. Unusually high dining spend
    anomaly_date = start_date + timedelta(days=28)
    amt_party = 6800.00
    balance -= amt_party
    transactions.append((anomaly_date, "UPI-SWIGGY-OFFICE-TEAM-LUNCH", amt_party, balance, "Debit"))

    # 2. Duplicate charge on consecutive days
    dup_date = start_date + timedelta(days=52)
    amt_dup = 1499.00
    transactions.append((dup_date, "AMAZON PRIME ANNUAL PLAN", amt_dup, balance - amt_dup, "Debit"))
    transactions.append((dup_date + timedelta(days=1), "AMAZON PRIME ANNUAL PLAN", amt_dup, balance - (2 * amt_dup), "Debit"))

    # Sort chronologically
    transactions.sort(key=lambda x: x[0])

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Description", "Amount", "Balance", "Type"])
        for date, desc, amount, bal, typ in transactions:
            writer.writerow([
                date.strftime("%d/%m/%Y"),
                desc,
                f"{amount:.2f}",
                f"{bal:.2f}",
                typ,
            ])

    print(f"Generated {len(transactions)} transactions over {months} months")
    print(f"Date range: {start_date.strftime('%d/%m/%Y')} — {end_date.strftime('%d/%m/%Y')}")
    print(f"Output: {output_file}")

    credits = sum(t[2] for t in transactions if t[4] == "Credit")
    debits = sum(t[2] for t in transactions if t[4] == "Debit")
    savings_rate = ((credits - debits) / credits) * 100 if credits > 0 else 0
    print(f"\nTotal Credits (Income): ₹{credits:,.2f}")
    print(f"Total Debits (Spend):   ₹{debits:,.2f}")
    print(f"Net Savings:            ₹{credits - debits:,.2f} ({savings_rate:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample bank statement CSV")
    parser.add_argument("--months", type=int, default=3, help="Number of months of data (default: 3)")
    parser.add_argument("--output", type=str, default="sample_statement.csv", help="Output filename")
    args = parser.parse_args()
    generate_statement(months=args.months, output_file=args.output)
