# 💱 USD/INR Forecast Telegram Bot

A Telegram bot that predicts USD/INR exchange rates across 9 time horizons using Facebook's Prophet ML model — with buy/sell signals and price charts.

---

## 🤖 Features

- 📊 Forecasts for 9 time horizons (10 mins → 1 week)
- 🟢 Buy / 🔴 Sell / ⚪ Hold signals
- 📈 Dark-themed price charts with confidence bands
- 🎯 Inline buttons to pick a specific horizon
- ⚡ Quick short-term signals (10m, 30m, 1hr)
- 🔐 Secure token handling via environment variables

---

## 📌 Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/predict` | Full forecast for all 9 horizons |
| `/quick` | Short-term signals (10m, 30m, 1hr) |
| `/choose` | Pick a specific time horizon |
| `/help` | Signal legend & info |

---

## 🧠 Signal Logic

| Signal | Condition |
|---|---|
| 🟢 STRONG BUY | Predicted +0.08% or more |
| 🟩 BUY | Predicted +0.03% to +0.08% |
| ⚪ HOLD | Low movement or high uncertainty |
| 🟥 SELL | Predicted -0.03% to -0.08% |
| 🔴 STRONG SELL | Predicted -0.08% or more |

---

## 🛠️ Tech Stack

- [Prophet](https://facebook.github.io/prophet/) — Time series forecasting
- [yfinance](https://pypi.org/project/yfinance/) — Live USD/INR data
- [python-telegram-bot](https://python-telegram-bot.org/) — Telegram bot framework
- [Matplotlib](https://matplotlib.org/) — Price charts
- [Railway](https://railway.app/) — 24/7 cloud hosting

---

## 🚀 Deployment

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/usd-inr-bot.git
cd usd-inr-bot
2. Install dependencies
pip install -r requirements.txt
3. Set environment variable
Create a .env file:
BOT_TOKEN=your_telegram_bot_token_here
4. Run locally
python main.py
5. Deploy to Railway
Push to GitHub
Connect repo on railway.app
Add BOT_TOKEN in Railway Variables tab
Railway auto-deploys on every push ✅
📁 Project Structure
usd-inr-bot/
├── main.py            # Main bot + model code
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignores .env file
└── README.md          # This file
⚠️ Disclaimer
This bot is for informational purposes only and does not constitute financial advice. Always do your own research before making any trading decisions.
👤 Author
Created by [jerin\_7575](https://www.instagram.com/jerin_7575?igsh=MWVuZ211MHluOXFsNA==)
