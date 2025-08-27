from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, get_flashed_messages, send_from_directory
import threading
import pandas as pd
import io
import json
import requests
import time
import logging
import smtplib
from datetime import datetime, timedelta
import secrets
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import werkzeug
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Float
from sqlalchemy.exc import IntegrityError
import re

# Your original imports
try:
    from producer import run_producer, stop_stream as producer_stop
    from consumer import consume_claims, stop_stream as consumer_stop, clear_queue
    from predictor import predict_fraud, load_model
except ImportError as e:
    logging.warning(f"Could not import stream components: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = '3f8c2e7a9b4d12f7c6a8e9f1b2d3c4e5f6a7b8c9d0e1f2a3b4c5d6e7f8g9h0i1'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)

# --- Models ---
class User(UserMixin, db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True, nullable=False)
    email: Mapped[str] = mapped_column(unique=True, nullable=False)
    password: Mapped[str] = mapped_column(unique=False, nullable=False)

class RecoveryCode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    code = db.Column(db.String(50), nullable=False, unique=True)
    expiry_time = db.Column(db.DateTime, nullable=False)
    user = db.relationship('User', back_populates='recovery_codes')

class EmailVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    code = db.Column(db.String(50), nullable=False)
    expiry_time = db.Column(db.DateTime, nullable=False)

User.recovery_codes = db.relationship('RecoveryCode', back_populates='user', lazy=True)

# --- Helper Functions ---
def send_verification_email(to_email, code):
    from_email = "noreplymedicaredesk@gmail.com"
    from_password = "ngjuvrcllvkvqrzm"
    subject = 'Email Verification Code - MediFetch'
    body = f"""
Hello,

Your verification code is: **{code}**

It will expire in 15 minutes.

If you did not request this, please ignore this email.

Thank you,
MediFetch Team
    """.strip()

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
            print(f"✅ Verification email sent to {to_email}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

def send_recovery_email(to_email, code):
    from_email = "noreplymedicaredesk@gmail.com"
    from_password = "ngjuvrcllvkvqrzm"
    subject = 'Password Recovery Code - MediFetch'
    body = f"""
Your recovery code is: **{code}**

It will expire in 15 minutes.

If you didn't request this, ignore this email.

Thank you,
MediFetch Team
    """.strip()

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
            print(f"✅ Recovery email sent to {to_email}")
    except Exception as e:
        print(f"❌ Error sending recovery email: {e}")

# --- Load Model for Batch Processing ---
try:
    load_model()
    logging.info("✅ Fraud detection model and explainer loaded successfully.")
except Exception as e:
    logging.error(f"🔴 WARNING: Could not load model. Batch processing will fail. Error: {e}")

# --- Groq API Config ---
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Routes ---
@login_manager.user_loader
def load_user(user_id):
    return db.get_or_404(User, user_id)

@app.route('/')
def index():
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        step = request.form.get("step")

        if step == "1":
            username = request.form.get("username")
            email = request.form.get("email")

            if not username or not email:
                flash("Username and email are required.", "error")
                return render_template("register.html", username=username, email=email)

            if User.query.filter_by(username=username).first():
                flash("Username already exists.", "error")
                return render_template("register.html", username=username, email=email)

            if User.query.filter_by(email=email).first():
                flash("Email already registered.", "error")
                return render_template("register.html", username=username, email=email)

            # Generate code
            code = secrets.token_hex(4)
            expiry_time = datetime.utcnow() + timedelta(minutes=15)

            # Remove old pending verifications
            EmailVerification.query.filter_by(email=email).delete()
            db.session.commit()

            # Save new verification
            entry = EmailVerification(username=username, email=email, code=code, expiry_time=expiry_time)
            db.session.add(entry)
            db.session.commit()

            send_verification_email(email, code)
            flash("A verification code has been sent to your email.", "success")
            return render_template("register.html", show_step2=True, email=email, username=username)

        elif step == "2":
            username = request.form.get("username")
            email = request.form.get("email")
            verification_code = request.form.get("verification_code")
            password = request.form.get("password")

            entry = EmailVerification.query.filter_by(email=email, code=verification_code).first()
            if not entry or entry.expiry_time < datetime.utcnow():
                flash("Invalid or expired verification code.", "error")
                return render_template("register.html", show_step2=True, email=email, username=username)

            if len(password) < 8:
                flash("Password must be at least 8 characters long.", "error")
                return render_template("register.html", show_step2=True, email=email, username=username)
            if not re.search(r'\d', password):
                flash("Password must contain at least one number.", "error")
                return render_template("register.html", show_step2=True, email=email, username=username)
            if not re.search(r'[A-Z]', password):
                flash("Password must contain at least one uppercase letter.", "error")
                return render_template("register.html", show_step2=True, email=email, username=username)

            # Create user
            hash_password = generate_password_hash(password, method='pbkdf2:sha256:600000', salt_length=8)
            user = User(username=username, email=email, password=hash_password)
            db.session.add(user)
            db.session.delete(entry)
            db.session.commit()

            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html", show_step2=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if not user:
            user = User.query.filter_by(username=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("secretz"))
        else:
            flash("Invalid email or password.", "error")

    return render_template("login.html")

@app.route('/forgetpassword', methods=['GET', 'POST'])
def forgetpassword():
    if request.method == "POST":
        email = request.form.get("email")
        user = User.query.filter_by(email=email).first()
        if user:
            code = secrets.token_hex(4)
            expiry_time = datetime.utcnow() + timedelta(minutes=15)
            recovery_code = RecoveryCode(user_id=user.id, code=code, expiry_time=expiry_time)
            db.session.add(recovery_code)
            db.session.commit()
            send_recovery_email(user.email, code)
            flash(f"A recovery code has been sent to {email}.", "success")
            return redirect(url_for('reset_password'))
        else:
            flash("Email not found.", "error")
    return render_template("forget.html")

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form.get("email")
        code = request.form.get("code")
        new_password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Invalid email.", "error")
            return redirect(url_for('reset_password'))

        recovery_code = RecoveryCode.query.filter_by(user_id=user.id, code=code).first()
        if not recovery_code or recovery_code.expiry_time < datetime.utcnow():
            flash("Invalid or expired code.", "error")
            return redirect(url_for('reset_password'))

        if len(new_password) < 8 or not any(c.isdigit() for c in new_password) or not any(c.isupper() for c in new_password):
            flash("Password must be 8+ chars with uppercase and number.", "error")
            return redirect(url_for('reset_password'))

        if check_password_hash(user.password, new_password):
            flash("New password cannot be the same as old one.", "error")
            return redirect(url_for('reset_password'))

        user.password = generate_password_hash(new_password, method='pbkdf2:sha256:600000', salt_length=8)
        db.session.delete(recovery_code)
        db.session.commit()

        flash("Password updated successfully!", "success")
        return redirect(url_for('login'))
    return render_template("reset_password.html")

@app.route('/secret', methods=['GET', 'POST'])
@login_required
def secretz():
    messages = get_flashed_messages(with_categories=True)
    return render_template("inference.html", username=current_user.username)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route('/download', methods=['GET'])
@login_required
def download():
    return send_from_directory('static', 'Drug report.pdf')

@app.route('/stream_page')
def stream_page():
    return render_template('stream.html')

@app.route('/batch_page')
def batch_page():
    return render_template('batch.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global producer_thread
    clear_queue()
    producer_thread = threading.Thread(target=run_producer)
    producer_thread.start()
    return {"status": "stream_started"}

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global producer_stop, consumer_stop
    producer_stop = True
    consumer_stop = True
    clear_queue()
    return {"status": "stream_stopped"}

@app.route('/stream')
def stream():
    def event_stream():
        for claim_json in consume_claims():
            yield claim_json
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Load model artifacts to get the list of required features
        artifacts = load_model()
        required_features = artifacts['features']

        # Check for missing features in the input data
        missing_cols = [col for col in required_features if col not in data]
        if 'InscClaimAmtReimbursed' not in data:
            missing_cols.append('InscClaimAmtReimbursed')
        
        if missing_cols:
            return jsonify({"error": f"Missing required features: {', '.join(missing_cols)}"}), 400

        # Create a DataFrame from the single data point
        # Ensure the DataFrame has the columns in the correct order for the model
        df = pd.DataFrame([data])
        df = df[required_features]
        
        result = predict_fraud(df)
        provider_id = data.get("Provider", "Unknown")
        claim_type = "Inpatient" if data.get('is_inpatient', 0) > 0 else "Outpatient"
        potential_savings = float(data.get('InscClaimAmtReimbursed', 0)) if result['risk_level'] in ['High', 'Medium'] else 0
        result.update({"provider_id": provider_id, "claim_type": claim_type, "potential_savings": potential_savings})
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in /predict_single: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        elif file.filename.endswith('.json'):
            df = pd.DataFrame(json.load(file.stream))
        else:
            return jsonify({"error": "Invalid file type. Use CSV or JSON."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {e}"}), 500

    try:
        artifacts = load_model()
        selected_features = artifacts['features']
        if 'InscClaimAmtReimbursed' not in df.columns:
            return jsonify({"error": "Missing 'InscClaimAmtReimbursed' column."}), 400
        missing_cols = [col for col in selected_features if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required features: {missing_cols}"}), 400

        results = []
        total_potential_savings = 0.0
        for _, row in df.iterrows():
            row_df = pd.DataFrame([row])
            pred = predict_fraud(row_df)
            claim_type = "Inpatient" if row.get('is_inpatient', 0) > 0 else "Outpatient"
            savings = row['InscClaimAmtReimbursed'] if pred['risk_level'] in ['High', 'Medium'] else 0
            total_potential_savings += savings
            results.append({
                "provider_id": row.get("Provider", "Unknown"),
                "potential_savings": savings,
                "claim_type": claim_type,
                **pred
            })

        summary = {
            "total_claims_processed": len(df),
            "total_fraud_flags": sum(1 for r in results if r['prediction'] == 'Fraud'),
            "total_potential_savings": total_potential_savings
        }
        return jsonify({"results": results, "summary": summary})
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/groq-explain', methods=['POST'])
def groq_explain():
    try:
        data = request.get_json()
        prompt = f"""
        Explain why this claim was flagged:
        Prediction: {data.get('prediction')}
        Risk Level: {data.get('risk_level')}
        Fraud Probability: {data.get('fraud_probability'):.2%}
        Provider ID: {data.get('provider_id')}
        Potential Savings: ${data.get('potential_savings'):,.2f}
        Key Factors: {data.get('explanation', 'N/A')}
        Keep it 2-3 sentences for investigators.
        """
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a fraud analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return jsonify({"explanation": response.json()["choices"][0]["message"]["content"].strip()})
    except Exception:
        return jsonify({"explanation": "Could not generate explanation."}), 500


@app.route('/claim')
def claim_form():
    return render_template('claim.html')


@app.route('/api/claim/analyze', methods=['POST'])
def analyze_claim():
    try:
        data = request.get_json()

        # Parse all 10 numeric fields from the form
        features = {
            'is_inpatient': float(data['is_inpatient']),
            'is_groupcode': float(data['is_groupcode']),
            'ChronicCond_rheumatoidarthritis': float(data['ChronicCond_rheumatoidarthritis']),
            'Beneficiaries_Count': float(data['Beneficiaries_Count']),
            'DeductibleAmtPaid': float(data['DeductibleAmtPaid']),
            'InscClaimAmtReimbursed': float(data['InscClaimAmtReimbursed']),
            'ChronicCond_Alzheimer': float(data['ChronicCond_Alzheimer']),
            'ChronicCond_IschemicHeart': float(data['ChronicCond_IschemicHeart']),
            'Days_Admitted': float(data['Days_Admitted']),
            'ChronicCond_stroke': float(data['ChronicCond_stroke']),
        }

        # 🔮 ML Model Logic (Replace this with your real model if available)
        import random
        fraud_probability = random.uniform(0.1, 0.95)
        risk_level = "High" if fraud_probability > 0.8 else "Medium" if fraud_probability > 0.5 else "Low"
        potential_savings = int(fraud_probability * 12000) if risk_level in ["High", "Medium"] else 0

        # 🤖 Call Groq API for AI-generated explanation
        explanation = get_groq_explanation(features, risk_level, fraud_probability, potential_savings)

        return jsonify({
            "risk_level": risk_level,
            "fraud_probability": round(fraud_probability, 4),
            "potential_savings": potential_savings,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def get_groq_explanation(features, risk_level, fraud_probability, potential_savings):
    prompt = f"""
    You are an AI Medicare fraud analyst. Analyze this claim:

    - Inpatient: {features['is_inpatient']}
    - Group Code: {features['is_groupcode']}
    - Chronic Conditions: 
      - Alzheimer: {features['ChronicCond_Alzheimer']}
      - Ischemic Heart: {features['ChronicCond_IschemicHeart']}
      - Stroke: {features['ChronicCond_stroke']}
      - Rheumatoid Arthritis: {features['ChronicCond_rheumatoidarthritis']}
    - Beneficiaries Count: {features['Beneficiaries_Count']}
    - Deductible Paid: ${features['DeductibleAmtPaid']:,.2f}
    - Claim Reimbursed: ${features['InscClaimAmtReimbursed']:,.2f}
    - Days Admitted: {features['Days_Admitted']}

    Risk Level: {risk_level}
    Confidence: {fraud_probability:.1%}
    Potential Savings if Fraud: ${potential_savings:,.2f}

    Provide a concise, professional explanation of why this claim is flagged as {risk_level} risk.
    Highlight key contributing factors and suggest next steps for investigators.
    Keep  and i need the prevention and also a detailed report to have a good understanding of the claims and recommended
    measure to prevent it.
    """

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.7
            }
        )
        res = response.json()
        return res['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"AI explanation unavailable (Groq API error: {str(e)})"

# Add to app.py

from predictor1 import predict_claim_fraud  # Ensure this matches your file

@app.route('/claim_batch_analyze', methods=['POST'])
def claim_batch_analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
        elif file.filename.endswith('.json'):
            df = pd.DataFrame(json.load(file.stream))
        else:
            return jsonify({"error": "Invalid file type. Use CSV or JSON."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {e}"}), 500

    try:
        results, summary = predict_claim_fraud(df)

        # Optional: Add AI insights
        for result in results:
            result['Amount'] = float(result.get('Amount', 0))
            result['confidence'] = result['fraud_probability']

        return jsonify({"results": results, "summary": summary})
    except Exception as e:
        app.logger.error(f"Claim batch error: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route('/claim_groq_summary', methods=['POST'])
def claim_groq_summary():
    try:
        data = request.get_json()
        prompt = f"""
        Summarize this claim fraud report:
        - Total claims: {data.get('total_claims_processed', 0)}
        - Fraud flags: {data.get('total_fraud_flags', 0)}
        - Potential savings: ${data.get('total_potential_savings', 0):,.2f}
        Top fraud drivers: {', '.join(data.get('top_fraud_drivers', ['N/A']))}
        Provide a concise summary and 3 actionable recommendations for CMS investigators.
        """
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a Medicare fraud analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(GROQ_API_URL.strip(), json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return jsonify({"explanation": response.json()["choices"][0]["message"]["content"].strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/groq-explain-summary', methods=['POST'])
def groq_explain_summary():
    try:
        data = request.get_json()
        prompt = f"""
        Summarize this fraud report for executives:
        - Total claims: {data.get('total_claims_processed', 0)}
        - Fraud flags: {data.get('total_fraud_flags', 0)}
        - Potential savings: ${data.get('total_potential_savings', 0):,.2f}
        Top drivers: {', '.join(data.get('top_fraud_drivers', ['N/A']))}
        Provide a concise summary and 3 actionable recommendations.
        """
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a fraud detection analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return jsonify({"explanation": response.json()["choices"][0]["message"]["content"].strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/claim_groq_explain', methods=['POST'])
def claim_groq_explain():
    try:
        data = request.get_json()
        prompt = f"""
        Explain why this claim was flagged as {data.get('risk_level')} risk:
        - Fraud Probability: {data.get('fraud_probability'):.1%}
        - Claim ID: {data.get('ClaimID')}
        - Potential Savings: ${data.get('potential_savings'):,.2f}
        - Key Factors: {data.get('explanation', 'N/A')}
        Provide a 2-3 sentence explanation for investigators and suggest prevention.
        """
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 200
        }
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(GROQ_API_URL.strip(), json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return jsonify({"explanation": response.json()["choices"][0]["message"]["content"].strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_claim_single', methods=['POST'])
def predict_claim_single():
    try:
        data = request.get_json()
        if not data:  # ← Fixed: Check if data is None or empty
            return jsonify({"error": "No data provided"}), 400

        # Validate required features
        required_features = {
            'Total_Claims_Per_Bene',
            'TimeInHptal',
            'Provider_Claim_Frequency',
            'ChronicCond_stroke_Yes',
            'DeductibleAmtPaid',
            'NoOfMonths_PartBCov',
            'NoOfMonths_PartACov',
            'OPD_Flag_Yes',
            'Diagnosis_Count',
            'ChronicDisease_Count',
            'Age'
        }

        missing = required_features - set(data.keys())
        if missing:
            return jsonify({"error": f"Missing required claim features: {sorted(missing)}"}), 400

        # Use claim-level model
        from predictor1 import predict_claim_fraud
        df = pd.DataFrame([data])
        results, _ = predict_claim_fraud(df)
        result = results[0]  # Single result

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error in /predict_claim_single: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
@app.route('/home')
def home():
    return render_template("inference.html")

@app.route('/inference_page')
@login_required
def inference_page():
    return render_template("claim.html")


# Route for stream1
@app.route('/stream1')
def stream1():
    return render_template("stream1.html")

# Route for batch1
@app.route('/batch1')
def batch1():
    return render_template("batch1.html")

# Route for claim1
@app.route('/claim1')
def claim1():
    return render_template("claim1.html")

    
@app.route('/claim_provider')
@login_required
def claim_provider():
    return render_template('claim.html')  # ✅ Provider-level form

@app.route('/claim_patient')
@login_required
def claim_patient():
    return render_template('claim1.html')  # ✅ Claim-level form

# Create tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
