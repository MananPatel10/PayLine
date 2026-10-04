"""
tax_optimizer.py — Institutional Automated Tax Optimization Engine
Compliant with Indian Income Tax Act (FY 2024-25 / AY 2025-26, Finance Act 2024).

Features:
1. Automated Deduction Harvesting from Statement (Rent/HRA, Insurance, Investments, Medical)
2. Dual-Regime Comparative Modeling (Old vs New Section 115BAC)
3. HRA Exemption Algorithm (Section 10(13A) Rule 2A)
4. Deduction Utilization & Headroom Gap Analysis (80C, 80D, 80CCD(1B), 24b)
5. Tactical Optimization Roadmap & Micro-Investing ELSS Sweep Integration
"""

import re
import numpy as np
import pandas as pd

def normalize_text(text: str) -> str:
    return str(text).upper().strip()

def harvest_tax_telemetry(df: pd.DataFrame, analysis_data: dict) -> dict:
    """
    Scans the bank statement DataFrame to harvest tax-relevant transactions:
    Rent payments, insurance premiums, investments, and salary inflow.
    """
    harvested = {
        "salary_transactions": [],
        "monthly_salary": 0.0,
        "annual_gross_income": 0.0,
        "rent_transactions": [],
        "monthly_rent": 0.0,
        "annual_rent_paid": 0.0,
        "insurance_80c": [],
        "insurance_80d": [],
        "investments_80c": [],
        "nps_80ccd": [],
        "loan_repayments": [],
        "detected_80c_total": 0.0,
        "detected_80d_total": 0.0,
        "detected_nps_total": 0.0,
        "detected_home_loan_interest": 0.0,
        "total_harvested_deductions": 0.0
    }

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        # Check if transactions list is passed directly or inside analysis_data
        txns = analysis_data.get("transactions") or []
        if not txns and isinstance(df, list):
            txns = df
        if txns:
            df = pd.DataFrame(txns)
            # Ensure standard column naming
            for old_col, new_col in [("date", "Date"), ("description", "Description"), ("amount", "Amount"), ("type", "Type"), ("category", "Category")]:
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            if "Type" not in df.columns:
                df["Type"] = "Debit"
        else:
            # Fallback to analysis summary if transactions are unavailable
            rent_spend = float(analysis_data.get("category_totals", {}).get("Rent", 0.0))
            if rent_spend > 0:
                harvested["monthly_rent"] = round(rent_spend, 2)
                harvested["annual_rent_paid"] = round(rent_spend * 12, 2)

            inc = float(analysis_data.get("income_vs_expense", {}).get("total_income", 0.0))
            harvested["annual_gross_income"] = round(inc * 12 if inc > 0 else 960000.0, 2)
            harvested["monthly_salary"] = round(harvested["annual_gross_income"] / 12, 2)

            estimated_epf = round(min(harvested["annual_gross_income"] * 0.06, 72000.0), 2)
            harvested["estimated_epf"] = estimated_epf
            harvested["detected_80c_total"] = estimated_epf
            return harvested

    # 1. Salary & Inflow Harvesting
    salary_mask = (df["Type"].str.lower() == "credit") & (
        (df["Category"].str.lower() == "salary") |
        (df["Description"].str.contains(r"SALARY|PAYROLL|STIPEND|NEFT-SALARY", case=False, regex=True))
    )
    sal_df = df[salary_mask]
    if not sal_df.empty:
        avg_monthly_sal = float(sal_df["Amount"].mean())
        harvested["monthly_salary"] = round(avg_monthly_sal, 2)
        harvested["annual_gross_income"] = round(avg_monthly_sal * 12, 2)
        sal_cols = [c for c in ["Date", "Description", "Amount"] if c in sal_df.columns]
        harvested["salary_transactions"] = sal_df[sal_cols].to_dict(orient="records")
    else:
        # Estimate from total credits
        total_cred = float(df[df["Type"].str.lower() == "credit"]["Amount"].sum())
        harvested["annual_gross_income"] = round(total_cred * 12, 2) if total_cred > 0 else 960000.0
        harvested["monthly_salary"] = round(harvested["annual_gross_income"] / 12, 2)

    # 2. Rent / HRA Harvesting
    rent_mask = (df["Type"].str.lower() == "debit") & (
        (df["Category"].str.lower() == "rent") |
        (df["Description"].str.contains(r"\b(?:RENT|HOUSE RENT|FLAT RENT|NOBROKER|RENTOMATION|TO OWNER)\b", case=False, regex=True))
    )
    rent_df = df[rent_mask]
    if not rent_df.empty:
        avg_rent = float(rent_df["Amount"].mean())
        harvested["monthly_rent"] = round(avg_rent, 2)
        harvested["annual_rent_paid"] = round(avg_rent * 12, 2)
        rent_cols = [c for c in ["Date", "Description", "Amount"] if c in rent_df.columns]
        harvested["rent_transactions"] = rent_df[rent_cols].to_dict(orient="records")

    # 3. Life Insurance / PPF / ELSS (Section 80C)
    life_mask = (df["Type"].str.lower() == "debit") & (
        df["Description"].str.contains(r"\b(?:LIC|HDFC LIFE|MAX LIFE|ICICI PRU|LIFE INS|TERM PLAN|PPF|ELSS|TAX SAVER)\b", case=False, regex=True)
    )
    life_df = df[life_mask]
    for _, row in life_df.iterrows():
        amt = float(row["Amount"])
        harvested["insurance_80c"].append({
            "date": str(row.get("Date", row.get("date", "Periodic"))),
            "description": str(row.get("Description", row.get("description", "Insurance"))),
            "amount": amt,
            "section": "80C"
        })
        harvested["detected_80c_total"] += amt

    # Standard EPF deduction heuristic: for corporate salaries, 12% of basic (~6% of gross) is typically deducted
    # If not explicitly in bank statement (since statements show net credited salary), note estimated EPF
    estimated_epf = round(min(harvested["annual_gross_income"] * 0.06, 72000.0), 2)
    harvested["estimated_epf"] = estimated_epf
    harvested["detected_80c_total"] += estimated_epf

    # 4. Health Insurance (Section 80D)
    health_mask = (df["Type"].str.lower() == "debit") & (
        df["Description"].str.contains(r"\b(?:STAR HEALTH|CARE HEALTH|NIVA BUPA|HEALTH INS|MEDICLAIM|MAX BUPA)\b", case=False, regex=True)
    )
    health_df = df[health_mask]
    for _, row in health_df.iterrows():
        amt = float(row["Amount"])
        harvested["insurance_80d"].append({
            "date": str(row.get("Date", row.get("date", "Periodic"))),
            "description": str(row.get("Description", row.get("description", "Health Insurance"))),
            "amount": amt,
            "section": "80D"
        })
        harvested["detected_80d_total"] += amt

    # 5. National Pension System (NPS Section 80CCD(1B))
    nps_mask = (df["Type"].str.lower() == "debit") & (
        df["Description"].str.contains(r"\b(?:NPS|CRA-NPS|PENSION FUND|NATIONAL PENSION)\b", case=False, regex=True)
    )
    nps_df = df[nps_mask]
    for _, row in nps_df.iterrows():
        amt = float(row["Amount"])
        harvested["nps_80ccd"].append({
            "date": str(row.get("Date", row.get("date", "Periodic"))),
            "description": str(row.get("Description", row.get("description", "NPS Contribution"))),
            "amount": amt,
            "section": "80CCD(1B)"
        })
        harvested["detected_nps_total"] += amt

    # 6. Home Loan Principal & Interest (Section 80C & Section 24b)
    loan_mask = (df["Type"].str.lower() == "debit") & (
        df["Description"].str.contains(r"\b(?:HOME LOAN|HOUSING LOAN|SBI HOME|HDFC HL)\b", case=False, regex=True)
    )
    loan_df = df[loan_mask]
    for _, row in loan_df.iterrows():
        amt = float(row["Amount"])
        # Standard amortization split: approx 40% principal (80C), 60% interest (24b)
        prin = round(amt * 0.40, 2)
        inte = round(amt * 0.60, 2)
        harvested["detected_80c_total"] += prin
        harvested["detected_home_loan_interest"] += inte
        harvested["loan_repayments"].append({
            "date": str(row.get("Date", row.get("date", "Periodic"))),
            "description": str(row.get("Description", row.get("description", "Home Loan EMI"))),
            "total_emi": amt,
            "principal_80c": prin,
            "interest_24b": inte
        })

    # Caps enforcement for harvested
    harvested["detected_80c_total"] = min(round(harvested["detected_80c_total"], 2), 150000.0)
    harvested["detected_80d_total"] = min(round(harvested["detected_80d_total"], 2), 25000.0)
    harvested["detected_nps_total"] = min(round(harvested["detected_nps_total"], 2), 50000.0)
    harvested["detected_home_loan_interest"] = min(round(harvested["detected_home_loan_interest"], 2), 200000.0)

    return harvested

def calculate_hra_exemption(annual_gross: float, annual_rent: float, is_metro: bool = True) -> dict:
    """
    Computes House Rent Allowance (HRA) exemption under Section 10(13A) Rule 2A.
    Exemption is the minimum of:
    1. Actual HRA received (assumed 40% of basic salary)
    2. Rent paid excess over 10% of basic salary
    3. 50% of basic salary for Metro (40% for non-metro)
    """
    # Industry convention: Basic Salary is ~50% of annual gross income
    basic_salary = annual_gross * 0.50
    estimated_hra_received = basic_salary * 0.40

    if annual_rent <= 0:
        return {
            "eligible_exemption": 0.0,
            "annual_rent_paid": 0.0,
            "basic_salary": round(basic_salary, 2),
            "estimated_hra_received": round(estimated_hra_received, 2),
            "rent_minus_ten_pct": 0.0,
            "metro_cap": round(basic_salary * (0.50 if is_metro else 0.40), 2),
            "status": "No Rent Detected"
        }

    condition_1 = estimated_hra_received
    condition_2 = max(0.0, annual_rent - (0.10 * basic_salary))
    condition_3 = basic_salary * (0.50 if is_metro else 0.40)

    exemption = min(condition_1, condition_2, condition_3)

    return {
        "eligible_exemption": round(exemption, 2),
        "annual_rent_paid": round(annual_rent, 2),
        "basic_salary": round(basic_salary, 2),
        "estimated_hra_received": round(condition_1, 2),
        "rent_minus_ten_pct": round(condition_2, 2),
        "metro_cap": round(condition_3, 2),
        "status": "Eligible" if exemption > 0 else "Below 10% Threshold"
    }

def compute_new_regime_tax(gross_income: float) -> dict:
    """
    Calculates tax under New Tax Regime (Section 115BAC, Finance Act 2024).
    Standard Deduction: INR 75,000.
    Section 87A Rebate: Zero tax if taxable income <= INR 7,00,000.
    Health & Education Cess: 4%.
    """
    std_deduction = 75000.0
    taxable_income = max(0.0, gross_income - std_deduction)

    # Section 87A rebate threshold
    if taxable_income <= 700000.0:
        return {
            "regime": "New Tax Regime (Sec 115BAC)",
            "gross_income": gross_income,
            "standard_deduction": std_deduction,
            "other_deductions": 0.0,
            "total_deductions": std_deduction,
            "taxable_income": taxable_income,
            "base_tax": 0.0,
            "rebate_87a": 0.0,
            "cess": 0.0,
            "total_tax": 0.0,
            "effective_tax_rate": 0.0,
            "rebate_applied": True,
            "slabs_breakdown": [
                {"slab": "Up to INR 3,00,000", "rate": "0%", "amount": 0.0},
                {"slab": "INR 3,00,001 - 7,00,000", "rate": "5%", "amount": 0.0}
            ]
        }

    # Tax slabs
    base_tax = 0.0
    slabs = []

    # 0 - 3L
    slabs.append({"slab": "Up to INR 3,00,000", "rate": "Nil", "amount": 0.0})

    # 3L - 7L (5% of up to 4L)
    if taxable_income > 300000.0:
        amt = min(taxable_income - 300000.0, 400000.0)
        t = amt * 0.05
        base_tax += t
        slabs.append({"slab": "INR 3,00,001 - 7,00,000", "rate": "5%", "amount": round(t, 2)})

    # 7L - 10L (10% of up to 3L)
    if taxable_income > 700000.0:
        amt = min(taxable_income - 700000.0, 300000.0)
        t = amt * 0.10
        base_tax += t
        slabs.append({"slab": "INR 7,00,001 - 10,00,000", "rate": "10%", "amount": round(t, 2)})

    # 10L - 12L (15% of up to 2L)
    if taxable_income > 1000000.0:
        amt = min(taxable_income - 1000000.0, 200000.0)
        t = amt * 0.15
        base_tax += t
        slabs.append({"slab": "INR 10,00,001 - 12,00,000", "rate": "15%", "amount": round(t, 2)})

    # 12L - 15L (20% of up to 3L)
    if taxable_income > 1200000.0:
        amt = min(taxable_income - 1200000.0, 300000.0)
        t = amt * 0.20
        base_tax += t
        slabs.append({"slab": "INR 12,00,001 - 15,00,000", "rate": "20%", "amount": round(t, 2)})

    # Above 15L (30%)
    if taxable_income > 1500000.0:
        amt = taxable_income - 1500000.0
        t = amt * 0.30
        base_tax += t
        slabs.append({"slab": "Above INR 15,00,000", "rate": "30%", "amount": round(t, 2)})

    cess = round(base_tax * 0.04, 2)
    total_tax = round(base_tax + cess, 2)
    effective_rate = round((total_tax / gross_income) * 100, 2) if gross_income > 0 else 0.0

    return {
        "regime": "New Tax Regime (Sec 115BAC)",
        "gross_income": gross_income,
        "standard_deduction": std_deduction,
        "other_deductions": 0.0,
        "total_deductions": std_deduction,
        "taxable_income": taxable_income,
        "base_tax": round(base_tax, 2),
        "rebate_87a": 0.0,
        "cess": cess,
        "total_tax": total_tax,
        "effective_tax_rate": effective_rate,
        "rebate_applied": False,
        "slabs_breakdown": slabs
    }

def compute_old_regime_tax(gross_income: float, deductions: float) -> dict:
    """
    Calculates tax under Old Tax Regime with standard deduction (INR 50,000)
    plus Chapter VI-A deductions (80C, 80D, 80CCD, HRA, 24b).
    Section 87A Rebate: Zero tax if taxable income <= INR 5,00,000.
    Health & Education Cess: 4%.
    """
    std_deduction = 50000.0
    total_deductions = round(std_deduction + max(0.0, deductions), 2)
    taxable_income = max(0.0, gross_income - total_deductions)

    if taxable_income <= 500000.0:
        return {
            "regime": "Old Tax Regime",
            "gross_income": gross_income,
            "standard_deduction": std_deduction,
            "other_deductions": round(deductions, 2),
            "total_deductions": total_deductions,
            "taxable_income": taxable_income,
            "base_tax": 0.0,
            "rebate_87a": 0.0,
            "cess": 0.0,
            "total_tax": 0.0,
            "effective_tax_rate": 0.0,
            "rebate_applied": True,
            "slabs_breakdown": [
                {"slab": "Up to INR 2,50,000", "rate": "0%", "amount": 0.0},
                {"slab": "INR 2,50,001 - 5,00,000", "rate": "5%", "amount": 0.0}
            ]
        }

    base_tax = 0.0
    slabs = []

    # 0 - 2.5L
    slabs.append({"slab": "Up to INR 2,50,000", "rate": "Nil", "amount": 0.0})

    # 2.5L - 5L (5% of 2.5L = 12,500)
    if taxable_income > 250000.0:
        amt = min(taxable_income - 250000.0, 250000.0)
        t = amt * 0.05
        base_tax += t
        slabs.append({"slab": "INR 2,50,001 - 5,00,000", "rate": "5%", "amount": round(t, 2)})

    # 5L - 10L (20% of up to 5L)
    if taxable_income > 500000.0:
        amt = min(taxable_income - 500000.0, 500000.0)
        t = amt * 0.20
        base_tax += t
        slabs.append({"slab": "INR 5,00,001 - 10,00,000", "rate": "20%", "amount": round(t, 2)})

    # Above 10L (30%)
    if taxable_income > 1000000.0:
        amt = taxable_income - 1000000.0
        t = amt * 0.30
        base_tax += t
        slabs.append({"slab": "Above INR 10,00,000", "rate": "30%", "amount": round(t, 2)})

    cess = round(base_tax * 0.04, 2)
    total_tax = round(base_tax + cess, 2)
    effective_rate = round((total_tax / gross_income) * 100, 2) if gross_income > 0 else 0.0

    return {
        "regime": "Old Tax Regime",
        "gross_income": gross_income,
        "standard_deduction": std_deduction,
        "other_deductions": round(deductions, 2),
        "total_deductions": total_deductions,
        "taxable_income": taxable_income,
        "base_tax": round(base_tax, 2),
        "rebate_87a": 0.0,
        "cess": cess,
        "total_tax": total_tax,
        "effective_tax_rate": effective_rate,
        "rebate_applied": False,
        "slabs_breakdown": slabs
    }

def analyze_tax(df: pd.DataFrame, analysis_data: dict, micro_data: dict = None) -> dict:
    """
    Primary tax optimization entrypoint:
    Runs full statement deduction harvesting, calculates HRA, evaluates Old vs New Tax Regimes,
    projects fully-optimized potential, and constructs actionable tax-reduction blueprints.
    """
    telemetry = harvest_tax_telemetry(df, analysis_data)
    gross_income = telemetry["annual_gross_income"]

    # 1. HRA computation
    hra_details = calculate_hra_exemption(gross_income, telemetry["annual_rent_paid"], is_metro=True)

    # 2. Harvested Deductions Total for Old Regime
    harvested_deductions = round(
        hra_details["eligible_exemption"] +
        telemetry["detected_80c_total"] +
        telemetry["detected_80d_total"] +
        telemetry["detected_nps_total"] +
        telemetry["detected_home_loan_interest"],
        2
    )

    # 3. Maximum Possible Optimization (Full utilization of 80C ₹1.5L, 80D ₹25k, NPS ₹50k + HRA)
    max_optimized_deductions = round(
        hra_details["eligible_exemption"] +
        150000.0 + # full 80C
        25000.0 +  # full 80D
        50000.0 +  # full NPS
        telemetry["detected_home_loan_interest"],
        2
    )

    # 4. Compute Tax Scenarios
    new_regime = compute_new_regime_tax(gross_income)
    old_regime_baseline = compute_old_regime_tax(gross_income, 0.0) # standard deduction only
    old_regime_harvested = compute_old_regime_tax(gross_income, harvested_deductions)
    old_regime_optimized = compute_old_regime_tax(gross_income, max_optimized_deductions)

    # 5. Regime Determination
    # Compare Current New Regime vs Current Harvested Old Regime
    if new_regime["total_tax"] < old_regime_harvested["total_tax"]:
        current_winner = "New Tax Regime"
        current_savings = round(old_regime_harvested["total_tax"] - new_regime["total_tax"], 2)
    elif old_regime_harvested["total_tax"] < new_regime["total_tax"]:
        current_winner = "Old Tax Regime"
        current_savings = round(new_regime["total_tax"] - old_regime_harvested["total_tax"], 2)
    else:
        current_winner = "Either (Tax is equal)"
        current_savings = 0.0

    # Compare New Regime vs Max Optimized Old Regime
    if old_regime_optimized["total_tax"] < new_regime["total_tax"]:
        potential_winner = "Old Tax Regime (Fully Optimized)"
        potential_savings_vs_new = round(new_regime["total_tax"] - old_regime_optimized["total_tax"], 2)
    else:
        potential_winner = "New Tax Regime"
        potential_savings_vs_new = 0.0

    # 6. Headroom & Gap Analysis
    c_80_unutilized = max(0.0, 150000.0 - telemetry["detected_80c_total"])
    d_80_unutilized = max(0.0, 25000.0 - telemetry["detected_80d_total"])
    nps_unutilized = max(0.0, 50000.0 - telemetry["detected_nps_total"])

    # Marginal tax rate for savings projection
    marginal_rate = 0.208 # 20% + 4% cess for standard middle tax bracket
    if gross_income > 1000000:
        marginal_rate = 0.312

    tax_shield_80c = round(c_80_unutilized * marginal_rate, 2)
    tax_shield_80d = round(d_80_unutilized * marginal_rate, 2)
    tax_shield_nps = round(nps_unutilized * marginal_rate, 2)

    # 7. Actionable Tactical Recommendations
    recommendations = []

    # HRA Recommendation
    if hra_details["eligible_exemption"] > 0:
        recommendations.append({
            "title": "Declare Section 10(13A) House Rent Allowance",
            "section": "Section 10(13A)",
            "impact": f"INR {hra_details['eligible_exemption']:,.2f} deduction",
            "tax_saved": round(hra_details["eligible_exemption"] * marginal_rate, 2),
            "urgency": "Immediate",
            "description": f"Payline detected INR {telemetry['annual_rent_paid']:,.2f} in annual rent payments. Ensure landlord PAN is submitted to payroll to claim this eligible HRA exemption."
        })

    # 80C ELSS Recommendation linked with Micro-investing
    if c_80_unutilized > 0:
        monthly_elss_needed = round(c_80_unutilized / 12, 2)
        recommendations.append({
            "title": "Deploy Spare Change into Section 80C ELSS Tax-Saver Funds",
            "section": "Section 80C",
            "impact": f"INR {c_80_unutilized:,.2f} remaining headroom",
            "tax_saved": tax_shield_80c,
            "urgency": "High",
            "description": f"Automate INR {monthly_elss_needed:,.2f}/month round-up sweeps into Equity Linked Savings Scheme (ELSS) mutual funds to exhaust your 80C cap while building long-term equity wealth."
        })

    # 80CCD NPS Recommendation
    if nps_unutilized > 0:
        recommendations.append({
            "title": "Utilize Exclusive Section 80CCD(1B) NPS Deduction",
            "section": "Section 80CCD(1B)",
            "impact": f"INR {nps_unutilized:,.2f} additional exemption",
            "tax_saved": tax_shield_nps,
            "urgency": "Medium",
            "description": "Contribute to Tier-1 National Pension System (NPS) for a dedicated tax deduction over and above the Section 80C limit."
        })

    # 80D Health Insurance
    if d_80_unutilized > 0:
        recommendations.append({
            "title": "Comprehensive Health Insurance Policy",
            "section": "Section 80D",
            "impact": f"INR {d_80_unutilized:,.2f} deduction cap",
            "tax_saved": tax_shield_80d,
            "urgency": "Medium",
            "description": "Insure yourself and dependent parents. Health insurance premiums up to INR 25,000 are deductible from gross taxable income."
        })

    # Summary payload
    return {
        "telemetry": telemetry,
        "gross_income": gross_income,
        "hra": hra_details,
        "harvested_deductions_total": harvested_deductions,
        "max_optimized_deductions": max_optimized_deductions,
        "new_regime": new_regime,
        "old_regime_baseline": old_regime_baseline,
        "old_regime_harvested": old_regime_harvested,
        "old_regime_optimized": old_regime_optimized,
        "current_winner": current_winner,
        "current_savings": current_savings,
        "potential_winner": potential_winner,
        "potential_savings_vs_new": potential_savings_vs_new,
        "headroom": {
            "c_80": {
                "utilized": telemetry["detected_80c_total"],
                "limit": 150000.0,
                "unutilized": c_80_unutilized,
                "pct_utilized": round((telemetry["detected_80c_total"] / 150000.0) * 100, 1),
                "potential_tax_shield": tax_shield_80c
            },
            "d_80": {
                "utilized": telemetry["detected_80d_total"],
                "limit": 25000.0,
                "unutilized": d_80_unutilized,
                "pct_utilized": round((telemetry["detected_80d_total"] / 25000.0) * 100, 1),
                "potential_tax_shield": tax_shield_80d
            },
            "nps_80ccd": {
                "utilized": telemetry["detected_nps_total"],
                "limit": 50000.0,
                "unutilized": nps_unutilized,
                "pct_utilized": round((telemetry["detected_nps_total"] / 50000.0) * 100, 1),
                "potential_tax_shield": tax_shield_nps
            }
        },
        "recommendations": recommendations
    }
