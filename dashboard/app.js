const API_URL = "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);

async function checkAPI() {
    const statusText = document.querySelector(".status");
    const dot = document.querySelector(".status-dot");

    try {
        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        statusText.innerHTML = `
            <span class="status-dot online"></span>
            API ONLINE
        `;
    } catch (error) {
        statusText.innerHTML = `
            <span class="status-dot offline"></span>
            API OFFLINE
        `;
    }
}


function getNumber(id) {
    return Number($(id).value);
}


async function analyzeTransaction() {

    const button = $("predictButton");

    button.disabled = true;
    button.textContent = "Analyzing...";

    const payload = {
        transaction_amount: getNumber("transaction_amount"),
        hour: getNumber("hour"),
        day_of_week: getNumber("day_of_week"),
        terminal_transaction_count: getNumber("terminal_transaction_count"),
        terminal_fraud_count: getNumber("terminal_fraud_count"),
        terminal_fraud_rate: getNumber("terminal_fraud_rate"),
        terminal_avg_amount: getNumber("terminal_avg_amount"),
        seconds_since_previous: getNumber("seconds_since_previous"),
        transactions_last_1h: getNumber("transactions_last_1h"),
        transactions_last_24h: getNumber("transactions_last_24h"),
        amount_last_1h: getNumber("amount_last_1h")
    };

    try {

        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();

        displayPrediction(data);

    } catch (error) {

        console.error(error);

        alert(
            "Unable to connect to MerchantGuard API.\n\n" +
            "Make sure Uvicorn is running on port 8000."
        );

    } finally {

        button.disabled = false;
        button.textContent = "Analyze Transaction Risk";
    }
}


function displayPrediction(data) {

    $("fraudProbability").textContent =
        `${(data.fraud_probability * 100).toFixed(2)}%`;

    $("riskScore").textContent =
        Number(data.risk_score).toFixed(2);

    $("riskLevel").textContent =
        data.risk_level;

    $("recommendedAction").textContent =
        data.recommended_action;

    $("fraudSignal").textContent =
        Number(data.signals.fraud_signal).toFixed(4);

    $("anomalySignal").textContent =
        Number(data.signals.anomaly_signal).toFixed(4);

    $("behaviorSignal").textContent =
        Number(data.signals.behavior_signal).toFixed(4);

    $("evidenceStrength").textContent =
        Number(data.signals.evidence_strength).toFixed(4);

    $("anomalyDetected").textContent =
        data.anomaly_detected ? "YES" : "NO";

    $("anomalyScore").textContent =
        Number(data.anomaly_score).toFixed(4);

    applyRiskStyle(data.risk_level);

    $("resultPanel").classList.add("result-visible");

    $("resultPanel").scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


function applyRiskStyle(level) {

    const riskElement = $("riskLevel");

    riskElement.classList.remove(
        "risk-low",
        "risk-medium",
        "risk-high",
        "risk-critical"
    );

    riskElement.classList.add(
        `risk-${level.toLowerCase()}`
    );
}


/* =========================================================
   ANALYTICS
========================================================= */

async function loadAnalytics() {

    try {

        const response = await fetch(
            `${API_URL}/analytics`
        );

        if (!response.ok) {
            throw new Error(
                `Analytics API returned ${response.status}`
            );
        }

        const data = await response.json();

        $("totalWindows").textContent =
            data.total_windows.toLocaleString();

        $("fraudWindows").textContent =
            data.fraud_windows.toLocaleString();

        $("highAlerts").textContent =
            data.high_alerts.toLocaleString();

        $("fraudAlerts").textContent =
            data.fraud_alerts.toLocaleString();

        $("alertPrecision").textContent =
            `${(
                data.alert_precision * 100
            ).toFixed(2)}%`;

        $("fraudCapture").textContent =
            `${(
                data.fraud_capture * 100
            ).toFixed(2)}%`;

        $("financialExposure").textContent =
            `₹${data.financial_exposure.toLocaleString(
                "en-IN",
                {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }
            )}`;

        const total = data.total_windows;

        setRiskBar(
            "lowCount",
            "lowBar",
            data.risk_distribution.LOW,
            total
        );

        setRiskBar(
            "mediumCount",
            "mediumBar",
            data.risk_distribution.MEDIUM,
            total
        );

        setRiskBar(
            "highCount",
            "highBar",
            data.risk_distribution.HIGH,
            total
        );

        setRiskBar(
            "criticalCount",
            "criticalBar",
            data.risk_distribution.CRITICAL,
            total
        );

    } catch (error) {

        console.error(
            "Unable to load analytics:",
            error
        );

    }
}

function setRiskBar(
    countId,
    barId,
    count,
    total
) {

    $(countId).textContent =
        count.toLocaleString();

    const percentage =
        (count / total) * 100;

    $(barId).style.width =
        `${Math.max(percentage, 0.5)}%`;

    $(barId).title =
        `${percentage.toFixed(2)}% of all windows`;
}

/* =========================================================
   TOP RISK ALERTS
========================================================= */

async function loadAlerts() {

    const tableBody = $("alertsTableBody");

    try {

        const response = await fetch(
            `${API_URL}/alerts`
        );

        if (!response.ok) {
            throw new Error(
                `Alerts API returned ${response.status}`
            );
        }

        const data = await response.json();

        tableBody.innerHTML = "";

        if (!data.alerts || data.alerts.length === 0) {

            tableBody.innerHTML = `
                <tr>
                    <td colspan="8">
                        No HIGH or CRITICAL alerts found.
                    </td>
                </tr>
            `;

            return;
        }

        data.alerts.forEach(alert => {

            const row = document.createElement("tr");

            const riskClass =
                alert.risk_level.toLowerCase();

            row.innerHTML = `
                <td>${alert.timestamp}</td>

                <td>${alert.terminal}</td>

                <td>
                    <span class="alert-risk ${riskClass}">
                        ${alert.risk_level}
                    </span>
                </td>

                <td>
                    ${Number(alert.risk_score).toFixed(2)}
                </td>

                <td>
                    ${Number(
                        alert.fraud_probability
                    ).toFixed(2)}%
                </td>

                <td>
                    ₹${Number(
                        alert.amount
                    ).toLocaleString(
                        "en-IN",
                        {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2
                        }
                    )}
                </td>

                <td>
                    ${alert.fraud_count}
                </td>

                <td>
                    ${alert.recommended_action}
                </td>
            `;

            tableBody.appendChild(row);
        });

    } catch (error) {

        console.error(
            "Unable to load alerts:",
            error
        );

        tableBody.innerHTML = `
            <tr>
                <td colspan="8">
                    Unable to load risk alerts.
                </td>
            </tr>
        `;
    }
}


/* =========================================================
   EVENT HANDLERS
========================================================= */

$("predictButton").addEventListener(
    "click",
    analyzeTransaction
);


/* =========================================================
   INITIALIZE DASHBOARD
========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        checkAPI();

        loadAnalytics();

        loadAlerts();

        setInterval(checkAPI, 10000);

    }
);