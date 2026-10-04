import os
import uuid
import json
import random
from datetime import datetime
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify, flash, session
from werkzeug.utils import secure_filename
import analyzer
import gemini_advisor
import tax_optimizer
import database

# Load local .env if present
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_file):
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'payline-enterprise-secret-key-2026')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# In-memory store for statement analytics and dataframes
DATA_STORE = {}

def process_statement(filepath: str, filename: str) -> dict:
    """
    Parses, normalizes, categorizes, and analyzes a bank statement CSV.
    Returns a unified data package.
    """
    df = analyzer.load_transactions(filepath)
    df = analyzer.categorize_transactions(df)
    analysis_raw = analyzer.analyze_spending(df)
    anomalies_raw = analyzer.detect_anomalies(df)
    recommendations_raw = analyzer.generate_recommendations(df, analysis_raw)
    micro_raw = analyzer.calculate_micro_investments(df, analysis_raw)
    tax_raw = tax_optimizer.analyze_tax(df, analysis_raw, micro_raw)
    subscriptions_raw = analyzer.detect_subscriptions(df)

    # Precompute visual category bar chart data with professional colors
    cat_totals = analysis_raw.get('category_totals', {})
    expense_cats = {k: v for k, v in cat_totals.items() if v > 0 and k not in ('Salary', 'Transfer')}
    max_spend = max(expense_cats.values()) if expense_cats else 1.0
    colors = ['#6366f1', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#64748b']
    
    category_breakdown = []
    for i, (cat, amt) in enumerate(sorted(expense_cats.items(), key=lambda x: x[1], reverse=True)):
        pct = min(100, max(5, round((amt / max_spend) * 100)))
        category_breakdown.append({
            'category': cat,
            'amount': amt,
            'pct': pct,
            'color': colors[i % len(colors)]
        })

    # Prepare normalized transaction objects for the ledger
    try:
        display_df = df.sort_values(by='Date', ascending=False)
    except Exception:
        display_df = df

    transactions = []
    for _, row in display_df.iterrows():
        d_val = row['Date']
        d_str = d_val.strftime('%d/%m/%Y') if hasattr(d_val, 'strftime') else str(d_val)
        conf_val = float(row.get('Confidence', 1.0)) * 100.0
        transactions.append({
            'date_str': d_str,
            'Description': str(row['Description']),
            'Category': str(row['Category']),
            'Amount': float(row['Amount']),
            'Type': str(row['Type']),
            'Confidence': round(conf_val, 1),
            'Source': str(row.get('Prediction_Source', 'Dual-Feature ML')),
        })

    # Prepare formatted anomaly entries
    anomalies = []
    for a in anomalies_raw:
        txn = a.get('transaction', {})
        date_val = txn.get('Date', '')
        if hasattr(date_val, 'strftime'):
            d_str = date_val.strftime('%d/%m/%Y')
        elif hasattr(date_val, 'date'):
            d_str = str(date_val.date())
        elif isinstance(date_val, str) and '00:00:00' in date_val:
            d_str = date_val.split(' ')[0]
        else:
            d_str = str(date_val) if date_val else 'Periodic'

        anomalies.append({
            'Date': d_str,
            'Description': str(txn.get('Description', 'Observed Debit')),
            'Amount': float(txn.get('Amount', 0)),
            'Category': str(txn.get('Category', 'Debit')),
            'Reason': str(a.get('explanation', '')),
            'Type': str(a.get('anomaly_type', 'Flagged Outlier')),
        })

    b_metrics = analysis_raw.get('budget_50_30_20', {
        'needs_spend': 0, 'needs_pct': 0, 'wants_spend': 0, 'wants_pct': 0, 'savings_pct': 0
    })

    return {
        'filename': filename,
        'row_count': len(df),
        'df': df,
        'analysis': analysis_raw,
        'b_metrics': b_metrics,
        'category_breakdown': category_breakdown,
        'anomalies': anomalies,
        'recommendations': recommendations_raw,
        'micro': micro_raw,
        'tax': tax_raw,
        'subscriptions': subscriptions_raw,
        'transactions': transactions,
        'dossier': None  # Lazy-loaded by Gemini advisor
    }

# Initialize Database and seed demo account with benchmark data
database.init_db()
database.seed_demo_user_if_needed(process_statement)

# Pre-load benchmark sample dataset if available on startup
SAMPLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_statement.csv')
if os.path.exists(SAMPLE_FILE):
    try:
        DATA_STORE['default'] = process_statement(SAMPLE_FILE, 'sample_statement.csv')
        print(f"Loaded benchmark dataset with {DATA_STORE['default']['row_count']} transactions.")
    except Exception as e:
        print(f"Could not pre-load sample statement: {e}")

@app.context_processor
def inject_user_context():
    """
    Makes current_user available to all Jinja2 templates.
    """
    user_id = session.get('user_id')
    if user_id:
        return {
            'current_user': {
                'id': user_id,
                'name': session.get('user_name', 'User'),
                'email': session.get('user_email', '')
            }
        }
    return {'current_user': None}

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please sign in to access your private financial analytics and ledger.', 'error')
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function

def get_active_data():
    """
    Returns (report_id, data_dict) for the authenticated user session.
    Unauthenticated users receive (None, None) to protect statement privacy.
    """
    user_id = session.get('user_id')
    if not user_id:
        return None, None

    # 1. Check in-memory session store for current report
    report_id = session.get('report_id')
    if report_id and report_id in DATA_STORE:
        return report_id, DATA_STORE[report_id]

    # 2. Look up user's latest statement from SQLite database
    db_report_id, db_data = database.get_latest_statement_for_user(user_id)
    if db_report_id and db_data:
        DATA_STORE[db_report_id] = db_data
        session['report_id'] = db_report_id
        return db_report_id, db_data

    return None, None

def get_active_meta(data):
    """
    Returns compact metadata dict for navbar display.
    """
    if not data:
        return None
    inc_exp = data['analysis'].get('income_vs_expense', {})
    return {
        'filename': data.get('filename', 'statement.csv'),
        'row_count': data.get('row_count', 0),
        'total_income': inc_exp.get('total_income', 0),
        'total_expense': inc_exp.get('total_expense', 0)
    }

# -----------------------------------------------------------------------------
# User Authentication Routes (Login, Signup, Logout)
# -----------------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))

    next_url = request.args.get('next') or request.form.get('next')

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        user = database.verify_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']

            # Connect user's statements from SQLite
            db_report_id, db_data = database.get_latest_statement_for_user(user['id'])
            if db_report_id and db_data:
                DATA_STORE[db_report_id] = db_data
                session['report_id'] = db_report_id
            else:
                session.pop('report_id', None)

            flash(f"Welcome back, {user['name']}!", 'success')
            if next_url and next_url.startswith('/'):
                return redirect(next_url)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password. Please verify your credentials.', 'error')

    _, data = get_active_data()
    return render_template('login.html', active_tab='login', active_dataset=get_active_meta(data), next_url=next_url)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not name or not email or len(password) < 6:
            flash('Please provide your name, valid email, and password of at least 6 characters.', 'error')
            return redirect(url_for('signup'))

        try:
            user = database.create_user(name, email, password)
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']
            session.pop('report_id', None)

            flash(f"Welcome to Payline, {name}! Your secure account has been created. Upload your statement to begin.", 'success')
            return redirect(url_for('upload'))
        except ValueError as e:
            flash(str(e), 'error')

    _, data = get_active_data()
    return render_template('signup.html', active_tab='signup', active_dataset=get_active_meta(data))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have successfully signed out.', 'success')
    return redirect(url_for('login'))

# -----------------------------------------------------------------------------
# Route 1: Home Page (Strictly Explanatory Landing Page - No Widgets)
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    _, data = get_active_data()
    return render_template('index.html', active_tab='home', active_dataset=get_active_meta(data))

# -----------------------------------------------------------------------------
# Route 2: Statement Ingestion Portal
# -----------------------------------------------------------------------------
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part provided in request.', 'error')
            return redirect(url_for('upload'))
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('upload'))

        if file and file.filename.lower().endswith('.csv'):
            try:
                filename = secure_filename(file.filename)
                report_id = str(uuid.uuid4())
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{report_id}_{filename}")
                file.save(filepath)

                # Process the statement
                data = process_statement(filepath, filename)
                DATA_STORE[report_id] = data
                session['report_id'] = report_id

                # Link statement to user's account in DB
                user_id = session.get('user_id')
                if user_id:
                    database.save_statement_for_user(user_id, report_id, filename, data['row_count'], data)

                flash(f"Successfully processed {data['row_count']} transactions from {filename}.", 'success')
                return redirect(url_for('dashboard'))
            except Exception as e:
                flash(f"Error parsing statement: {str(e)}", 'error')
                return redirect(url_for('upload'))
        else:
            flash('Only CSV format statements are supported.', 'error')
            return redirect(url_for('upload'))

    _, data = get_active_data()
    return render_template('upload.html', active_tab='upload', active_dataset=get_active_meta(data))

@app.route('/load-sample')
@login_required
def load_sample():
    user_id = session.get('user_id')
    if 'default' in DATA_STORE:
        session['report_id'] = 'default'
        if user_id:
            database.save_statement_for_user(
                user_id, 'sample_default', 'sample_statement.csv',
                DATA_STORE['default']['row_count'], DATA_STORE['default']
            )
        flash(f"Loaded verified benchmark statement ({DATA_STORE['default']['row_count']} transactions).", 'success')
        return redirect(url_for('dashboard'))
    elif os.path.exists(SAMPLE_FILE):
        try:
            DATA_STORE['default'] = process_statement(SAMPLE_FILE, 'sample_statement.csv')
            session['report_id'] = 'default'
            if user_id:
                database.save_statement_for_user(
                    user_id, 'sample_default', 'sample_statement.csv',
                    DATA_STORE['default']['row_count'], DATA_STORE['default']
                )
            flash(f"Loaded verified benchmark statement ({DATA_STORE['default']['row_count']} transactions).", 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f"Failed to load sample dataset: {e}", 'error')
            return redirect(url_for('upload'))
    else:
        flash('Sample dataset not found on disk.', 'error')
        return redirect(url_for('upload'))

# -----------------------------------------------------------------------------
# Route 3: Financial Health & Analytics Dashboard
# -----------------------------------------------------------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    report_id, data = get_active_data()
    if not data:
        flash('No bank statement loaded for your account yet. Please upload a CSV statement or load the benchmark dataset to generate your analytics.', 'error')
        return redirect(url_for('upload'))

    return render_template(
        'dashboard.html',
        active_tab='dashboard',
        active_dataset=get_active_meta(data),
        analysis=data['analysis'],
        b_metrics=data['b_metrics'],
        category_breakdown=data['category_breakdown'],
        anomalies=data['anomalies'],
        transactions=data['transactions']
    )

# -----------------------------------------------------------------------------
# Route 4: Micro-Investment Engine Hub
# -----------------------------------------------------------------------------
@app.route('/invest')
@login_required
def invest():
    report_id, data = get_active_data()
    if not data:
        flash('No bank statement loaded for your account yet. Please upload a CSV statement or load the benchmark dataset.', 'error')
        return redirect(url_for('upload'))

    # Lazy-compute or enrich micro analytics if not present or needs refresh
    if not data.get('micro') or not data['micro'].get('demat_holdings'):
        if data.get('df') is not None:
            data['micro'] = analyzer.calculate_micro_investments(data['df'], data['analysis'])
        elif data.get('transactions'):
            df_reconstructed = pd.DataFrame(data['transactions'])
            df_reconstructed['Date'] = pd.to_datetime(df_reconstructed['date_str'], format='%d/%m/%Y', errors='coerce')
            data['micro'] = analyzer.calculate_micro_investments(df_reconstructed, data['analysis'])

    user_id = session.get('user_id')
    executed_sweeps = database.get_user_investments(user_id) if user_id else []
    total_swept = sum(s['amount'] for s in executed_sweeps)

    return render_template(
        'invest.html',
        active_tab='invest',
        active_dataset=get_active_meta(data),
        micro=data['micro'],
        analysis=data['analysis'],
        tax=data.get('tax', {}),
        executed_sweeps=executed_sweeps,
        total_swept=round(total_swept, 2),
        portfolio_value=round(total_swept * 1.084, 2)
    )

@app.route('/api/invest/mandate', methods=['POST'])
@login_required
def api_invest_mandate():
    req_data = request.get_json(silent=True) or {}
    frequency = req_data.get('frequency', 'Daily Midnight Sweep')
    cap = float(req_data.get('monthly_cap', 15000.0))
    return jsonify({
        'status': 'success',
        'frequency': frequency,
        'monthly_cap': cap,
        'message': f'NPCI e-Mandate updated successfully to {frequency} with cap INR {cap:,.2f}'
    })

@app.route('/api/invest/sweep', methods=['POST'])
@login_required
def api_invest_sweep():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401

    req_data = request.get_json(silent=True) or {}
    amount = float(req_data.get('amount', 0.0))
    roundup_rule = req_data.get('rule', 'Nearest ₹50')
    allocation = req_data.get('allocation', 'balanced')
    payment_method = req_data.get('payment_method', 'UPI (Google Pay)')
    custom_utr = req_data.get('utr_number')

    if amount <= 0:
        return jsonify({'error': 'Investment amount must be greater than zero.'}), 400

    order_id = f"ORD-PAYLINE-{uuid.uuid4().hex[:8].upper()}"
    utr_number = custom_utr if custom_utr else f"4291{random.randint(10000000, 99999999)}"
    folio_number = f"FOLIO-98214-{random.randint(10, 99)}"
    nav = 289.45
    units_allotted = round(amount / nav, 3)
    stamp_duty = round(amount * 0.00005, 2)
    net_invested = round(amount - stamp_duty, 2)

    sms_alert = f"HDFC Bank Alert: Rs {amount:,.2f} debited from A/c **8912 on {datetime.now().strftime('%d-%b-%Y %H:%M')} IST to VPA payline.invest@icici. Ref/UTR: {utr_number}."

    receipt_data = {
        'order_id': order_id,
        'utr_number': utr_number,
        'folio_number': folio_number,
        'timestamp': datetime.now().strftime('%d/%m/%Y, %H:%M:%S IST'),
        'gross_amount': amount,
        'stamp_duty': stamp_duty,
        'net_invested': net_invested,
        'allotment_nav': nav,
        'units_allotted': units_allotted,
        'scheme': 'UTI Nifty 50 Index Fund - Direct Plan - Growth',
        'investor_name': session.get('user_name', 'Demo User'),
        'investor_email': session.get('user_email', 'demo@payline.com'),
        'investor_pan': 'ABCDE1234F',
        'demat_acc': '12081600-09824102 (CDSL India)',
        'payment_method': payment_method,
        'bank_ref': f"HDFC-UPI-{random.randint(100000, 999999)}",
        'amfi_reg': 'ARN-284910',
        'sms_alert': sms_alert
    }

    rec = database.record_investment(
        user_id=user_id,
        order_id=order_id,
        amount=amount,
        roundup_rule=roundup_rule,
        allocation=allocation if isinstance(allocation, dict) else {'strategy': allocation},
        payment_method=payment_method,
        utr_number=utr_number,
        folio_number=folio_number,
        units_allotted=units_allotted,
        nav=nav,
        receipt_data=receipt_data
    )

    all_sweeps = database.get_user_investments(user_id)
    total_swept = sum(s['amount'] for s in all_sweeps)

    return jsonify({
        'status': 'success',
        'order_id': order_id,
        'amount': amount,
        'roundup_rule': roundup_rule,
        'allocation': allocation,
        'payment_method': payment_method,
        'utr_number': utr_number,
        'folio_number': folio_number,
        'units_allotted': units_allotted,
        'nav': nav,
        'stamp_duty': stamp_duty,
        'sms_alert': sms_alert,
        'receipt': receipt_data,
        'total_swept': round(total_swept, 2),
        'portfolio_value': round(total_swept * 1.084, 2),
        'message': f'Successfully deployed INR {amount:,.2f} into automated wealth portfolio.'
    })

@app.route('/api/invest/receipt/<order_id>')
@login_required
def api_invest_receipt(order_id):
    user_id = session.get('user_id')
    sweeps = database.get_user_investments(user_id)
    for s in sweeps:
        if s['order_id'] == order_id:
            return jsonify({'status': 'success', 'receipt': s.get('receipt', {}), 'investment': s})
    return jsonify({'error': 'Order not found'}), 404

# -----------------------------------------------------------------------------
# Route 5: Automated Tax Optimization Suite
# -----------------------------------------------------------------------------
@app.route('/tax')
@login_required
def tax():
    report_id, data = get_active_data()
    if not data:
        flash('No bank statement loaded for your account yet. Please upload a CSV statement or load the benchmark dataset.', 'error')
        return redirect(url_for('upload'))

    # Lazy-compute tax analytics if not present in cached profile
    if not data.get('tax') or data.get('tax', {}).get('hra', {}).get('eligible_exemption') == 0:
        data['tax'] = tax_optimizer.analyze_tax(
            data.get('df') if data.get('df') is not None else data.get('transactions'),
            data['analysis'],
            data.get('micro')
        )
        user_id = session.get('user_id')
        if user_id:
            database.save_statement_for_user(
                user_id, report_id, data.get('filename', 'statement.csv'),
                data.get('row_count', 0), data
            )

    return render_template(
        'tax.html',
        active_tab='tax',
        active_dataset=get_active_meta(data),
        tax=data['tax'],
        analysis=data['analysis'],
        micro=data.get('micro', {})
    )

@app.route('/api/tax')
@login_required
def api_tax():
    report_id, data = get_active_data()
    if not data:
        return jsonify({'error': 'No active bank statement loaded.'}), 400
    if not data.get('tax') or data.get('tax', {}).get('hra', {}).get('eligible_exemption') == 0:
        data['tax'] = tax_optimizer.analyze_tax(
            data.get('df') if data.get('df') is not None else data.get('transactions'),
            data['analysis'],
            data.get('micro')
        )
    return jsonify(data['tax'])

# -----------------------------------------------------------------------------
# Route 6: Executive Gemini AI Advisor
# -----------------------------------------------------------------------------
@app.route('/advisor')
@login_required
def advisor():
    report_id, data = get_active_data()
    if not data:
        flash('No bank statement loaded for your account yet. Please upload a CSV statement or load the benchmark dataset.', 'error')
        return redirect(url_for('upload'))

    # Ensure tax analysis is ready for advisor context
    if not data.get('tax') or data.get('tax', {}).get('hra', {}).get('eligible_exemption') == 0:
        data['tax'] = tax_optimizer.analyze_tax(
            data.get('df') if data.get('df') is not None else data.get('transactions'),
            data['analysis'],
            data.get('micro')
        )

    # Lazy-load CFO dossier once per statement or upgrade if offline
    if not data.get('dossier') or data.get('dossier', {}).get('status') != 'online':
        new_dossier = gemini_advisor.generate_cfo_dossier(data['analysis'], data['micro'], data.get('tax'))
        data['dossier'] = new_dossier
        # Save back to database if logged in
        user_id = session.get('user_id')
        if user_id:
            database.save_statement_for_user(
                user_id, report_id, data.get('filename', 'statement.csv'),
                data.get('row_count', 0), data
            )

    dossier = data.get('dossier', {})
    if dossier and dossier.get('dossier_markdown'):
        dossier['dossier_html'] = gemini_advisor.render_markdown_html(dossier['dossier_markdown'])

    return render_template(
        'advisor.html',
        active_tab='advisor',
        active_dataset=get_active_meta(data),
        dossier=dossier,
        analysis=data['analysis'],
        micro=data['micro']
    )

# -----------------------------------------------------------------------------
# Route 7: Interactive Chat API Endpoint (Gemini 3.8 Flash)
# -----------------------------------------------------------------------------
@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    report_id, data = get_active_data()
    if not data:
        return jsonify({'error': 'No active bank statement loaded.'}), 400

    req_data = request.get_json(silent=True) or {}
    query = req_data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Query text is required.'}), 400

    # Ensure tax telemetry is passed to chatbot context
    if not data.get('tax'):
        data['tax'] = tax_optimizer.analyze_tax(data.get('df'), data['analysis'], data.get('micro'))

    response_text = gemini_advisor.query_gemini_advisor(
        user_query=query,
        analysis_data=data['analysis'],
        micro_data=data['micro'],
        tax_data=data['tax']
    )

    return jsonify({
        'status': 'success',
        'query': query,
        'response': response_text
    })

# -----------------------------------------------------------------------------
# Route 8: Recurring Subscriptions & Silent Cash Leakage Guard
# -----------------------------------------------------------------------------
@app.route('/subscriptions')
@login_required
def subscriptions():
    report_id, data = get_active_data()
    if not data:
        flash('No bank statement loaded for your account yet. Please upload a CSV statement or load the benchmark dataset.', 'error')
        return redirect(url_for('upload'))

    # Lazy-compute subscriptions if not present in cached profile
    if not data.get('subscriptions'):
        if data.get('df') is not None:
            data['subscriptions'] = analyzer.detect_subscriptions(data['df'])
        elif data.get('transactions'):
            data['subscriptions'] = analyzer.detect_subscriptions(data['transactions'])

    return render_template(
        'subscriptions.html',
        active_tab='subscriptions',
        active_dataset=get_active_meta(data),
        subs=data['subscriptions'],
        analysis=data['analysis'],
        micro=data.get('micro', {})
    )

@app.route('/api/subscriptions')
@login_required
def api_subscriptions():
    report_id, data = get_active_data()
    if not data:
        return jsonify({'error': 'No active bank statement loaded.'}), 400
    if not data.get('subscriptions'):
        if data.get('df') is not None:
            data['subscriptions'] = analyzer.detect_subscriptions(data['df'])
        elif data.get('transactions'):
            data['subscriptions'] = analyzer.detect_subscriptions(data['transactions'])
    return jsonify(data['subscriptions'])

# -----------------------------------------------------------------------------
# Route 9: Machine-Readable Analysis JSON API
# -----------------------------------------------------------------------------
@app.route('/api/analysis')
@login_required
def api_analysis():
    report_id, data = get_active_data()
    if not data:
        return jsonify({'error': 'No active bank statement loaded.'}), 404

    return jsonify({
        'status': 'success',
        'report_id': report_id,
        'filename': data.get('filename'),
        'row_count': data.get('row_count'),
        'analysis': data.get('analysis'),
        'anomalies': data.get('anomalies'),
        'micro_investments': data.get('micro'),
        'tax_optimization': data.get('tax'),
        'subscriptions': data.get('subscriptions')
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
