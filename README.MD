MerchantGuard AI
AI-Powered Merchant Fraud & Risk Detection

MerchantGuard AI is a defense-only AI risk management system designed to help merchants identify suspicious transaction behavior and high-risk merchant terminal activity.

It combines:

Supervised fraud detection
Unsupervised anomaly detection
Merchant behavioral analysis
Evidence strength
A unified risk scoring engine
Action-oriented risk recommendations
A FastAPI prediction service
A web-based merchant risk dashboard

The project is built for Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager.

Problem

Fraud does not always appear as a single obviously fraudulent transaction.

A merchant terminal can gradually exhibit suspicious behavior such as:

Increasing transaction frequency
Unusual transaction amounts
Repeated suspicious transactions
Abnormal activity during unusual hours
Sudden changes from historical merchant behavior

A useful risk system therefore needs to combine multiple signals instead of relying on a single fraud probability.

MerchantGuard AI addresses this by combining transaction-level fraud prediction with terminal-level anomaly and behavioral signals.

Solution

MerchantGuard AI analyzes transaction and merchant-terminal behavior through multiple complementary signals.

Transaction
     │
     ▼
Feature Engineering
     │
     ├───────────────┐
     ▼               ▼
Fraud Model     Anomaly Model
(Random Forest) (Isolation Forest)
     │               │
     └───────┬───────┘
             ▼
     Behavioral Analysis
             │
             ▼
      Evidence Strength
             │
             ▼
       MerchantGuard
        Risk Engine
             │
             ▼
      Risk Score 0–100
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
      LOW  MEDIUM HIGH/CRITICAL
             │
             ▼
      Recommended Action
AI Architecture
1. Supervised Fraud Detection

A Random Forest classifier estimates the probability that a transaction is fraudulent.

The model uses transaction, temporal, terminal-history and behavioral features including:

Transaction amount
Hour
Day of week
Log transaction amount
Terminal transaction count
Terminal fraud count
Terminal fraud rate
Terminal average amount
Amount deviation
Time since previous transaction
Transactions in the last hour
Transactions in the last 24 hours
Amount processed in the last hour
2. Unsupervised Anomaly Detection

An Isolation Forest identifies unusual terminal-hour behavior.

The anomaly model uses:

Transaction count
Total transaction amount
Average transaction amount

This allows MerchantGuard to detect unusual activity even when the behavior does not exactly match previously observed fraud patterns.

3. Behavioral Analysis

MerchantGuard compares current activity with historical merchant-terminal behavior.

Behavioral signals include:

Transaction volume deviation
Amount deviation
Historical terminal behavior
Merchant activity evidence
4. Risk Engine

The final risk score combines the major signals:

40% Fraud Signal
30% Anomaly Signal
20% Behavioral Signal
10% Evidence Strength

The score is then adjusted according to evidence strength and high-value transaction behavior.

Risk levels:

Score	Risk Level	Recommended Action
< 30	LOW	Normal monitoring
30–59	MEDIUM	Enhanced monitoring
60–79	HIGH	Additional verification
≥ 80	CRITICAL	Merchant review + enhanced verification
Dataset

MerchantGuard uses the simulated payment transaction dataset from the Fraud Detection Handbook.

The project currently uses a 30-day subset:

Period:       2018-04-01 → 2018-04-30
Transactions: 288,062
Fraud:        1,702
Fraud rate:   0.5908%
Customers:    4,959
Terminals:    10,000

The data is transformed into terminal-hour behavioral windows for merchant risk evaluation.

Evaluation Methodology

To avoid leaking future information into model evaluation, the data is split chronologically.

TRAIN
April 1 – April 20

        ↓

VALIDATION
April 21 – April 25

        ↓

HELD-OUT TEST
April 26 – April 30

The final MerchantGuard evaluation is performed on the held-out test period.

Held-out test results
Metric	Result
Terminal-hour windows	46,907
Fraud-containing windows	429
HIGH alerts	52
CRITICAL alerts	83
HIGH + CRITICAL alerts	135
Fraud-containing alerts	133
Alert precision	98.52%
Fraud-window capture	31.00%
Retrospective exposure	₹41,848.29
Interpretation

MerchantGuard's HIGH/CRITICAL alert set achieved:

98.52% precision

meaning that 133 of the 135 HIGH/CRITICAL alert windows contained at least one fraud transaction in the held-out test data.

The system captured:

31.00% of fraud-containing terminal-hour windows

while keeping the number of high-severity alerts relatively small.

The ₹41,848.29 exposure figure is a retrospective evaluation metric calculated from actual fraud labels in the held-out test set, not a live prediction of future losses.

Live Prediction API

MerchantGuard exposes a FastAPI service.

Health
GET /health
Transaction prediction
POST /predict

Example:

{
  "transaction_amount": 500,
  "hour": 2,
  "day_of_week": 2,
  "terminal_transaction_count": 20,
  "terminal_fraud_count": 3,
  "terminal_fraud_rate": 0.15,
  "terminal_avg_amount": 50,
  "seconds_since_previous": 10,
  "transactions_last_1h": 15,
  "transactions_last_24h": 80,
  "amount_last_1h": 5000
}

The API returns:

Fraud probability
Anomaly detection
Anomaly score
Risk score
Risk level
Recommended action
Individual risk signals
Historical analytics
GET /analytics

Provides:

Total evaluation windows
Fraud windows
HIGH/CRITICAL alerts
Fraud-containing alerts
Alert precision
Fraud capture
Retrospective exposure
Risk distribution
Top risk alerts
GET /alerts

Returns the highest-risk HIGH and CRITICAL terminal-hour windows.

Dashboard

MerchantGuard includes a browser-based risk dashboard.

The dashboard provides:

Transaction Analysis

Users can enter transaction and merchant-terminal information and receive an immediate risk assessment.

Risk Assessment

Displays:

Fraud probability
Risk score
Risk level
Recommended action
Detection Signals

Displays:

Fraud signal
Anomaly signal
Behavioral signal
Evidence strength
Anomaly detection
Anomaly score
Merchant Risk Analytics

Displays:

Total test windows
Fraud windows
HIGH/CRITICAL alerts
Fraud alerts
Alert precision
Fraud capture
Retrospective financial exposure
Risk distribution
Top Risk Alerts

Displays the highest-risk merchant terminal windows with:

Timestamp
Terminal
Risk level
Risk score
Fraud probability
Transaction amount
Fraud count
Recommended action
Example Risk Alert

Example output from the system:

Timestamp:          2018-04-28 10:00:00
Terminal:           3644
Risk Level:         CRITICAL
Risk Score:         90.18
Fraud Probability:  93.43%
Amount:             ₹182.27
Fraud Count:        2

Action:
Merchant review + enhanced verification
Technology Stack
Machine Learning
Python
pandas
NumPy
scikit-learn
Random Forest
Isolation Forest
joblib
Backend
FastAPI
Pydantic
Uvicorn
Frontend
HTML
CSS
JavaScript
Live Server
Data
Fraud Detection Handbook simulated transaction dataset
Project Structure
MerchantGuard-AI/
│
├── dashboard/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── data/
│   └── processed/
│
├── models/
│   ├── random_forest_fraud_model.joblib
│   ├── isolation_forest_model.joblib
│   ├── anomaly_scaler.joblib
│   └── fraud_features.txt
│
├── src/
│   ├── prepare_dataset.py
│   ├── features.py
│   ├── validate_features.py
│   ├── split_data.py
│   ├── fraud_model.py
│   ├── merchant_anomaly.py
│   ├── merchant_baseline.py
│   ├── merchantguard_engine.py
│   ├── evaluate_merchantguard.py
│   └── api.py
│
└── README.md
Running the Project
1. Create and activate virtual environment

Windows:

python -m venv venv

Git Bash:

source venv/Scripts/activate
2. Install dependencies
pip install -r requirements.txt
3. Start FastAPI
uvicorn src.api:app --reload

API:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
4. Start the dashboard

Open the dashboard folder with VS Code Live Server.

Dashboard:

http://127.0.0.1:5500/dashboard/index.html
Defense-Only Design

MerchantGuard AI is designed strictly for fraud detection, risk assessment and defensive intervention.

The system does not:

Generate fraudulent transactions
Bypass payment security
Exploit payment infrastructure
Automate unauthorized financial activity
Provide offensive fraud techniques

Its purpose is to help merchants identify suspicious behavior and apply appropriate defensive controls.

Future Improvements

Potential future improvements include:

Real-time transaction streaming
Merchant-specific adaptive thresholds
Online model updating
Cost-sensitive threshold optimization
False-positive cost optimization
Explainable AI using SHAP
Alert history and case management
Merchant-level risk trends
Production deployment
Integration with payment webhooks
Buildathon Track

Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager

MerchantGuard AI focuses on the track's fraud-loss problem by combining transaction-level fraud detection with merchant-terminal anomaly and behavioral risk analysis.

The implementation emphasizes:

A working AI detector
Held-out evaluation
Measured precision and recall/capture
Honest reporting of evaluation metrics
Defensive risk intervention
An operational dashboard

Razorpay's stated Track 2 bar specifically calls for measured performance on held-out data and honest metrics, including false-positive cost.




Author:
Yarlagadda Pavani