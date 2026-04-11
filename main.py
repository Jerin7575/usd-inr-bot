import os
import io
import nest_asyncio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf

from datetime import datetime, timedelta
from dotenv import load_dotenv
from prophet import Prophet
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# ── Load Token ─────────────────────────────────────────────────
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN environment variable not set!")

nest_asyncio.apply()

# ── Horizons ───────────────────────────────────────────────────
HORIZONS = {
    "10 mins":  "10min",
    "30 mins":  "30min",
    "1 hour":   "60min",
    "3 hours":  "180min",
    "6 hours":  "360min",
    "12 hours": "720min",
    "1 day":    "1440min",
    "3 days":   "4320min",
    "1 week":   "10080min",
}

OFFSET_MINS = {
    "10 mins": 10, "30 mins": 30, "1 hour": 60,
    "3 hours": 180, "6 hours": 360, "12 hours": 720,
    "1 day": 1440, "3 days": 4320, "1 week": 10080,
}

SIGNAL_COLORS = {
    "🟢 STRONG BUY":  "#00c853",
    "🟩 BUY":          "#69f0ae",
    "⚪ HOLD":          "#90a4ae",
    "🟥 SELL":          "#ff5252",
    "🔴 STRONG SELL": "#b71c1c",
}

# ── Data Fetching ──────────────────────────────────────────────
def get_usd_inr_data():
    ticker = yf.Ticker("INR=X")
    df = ticker.history(period="60d", interval="1h")
    df = df[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    return df

def get_current_price():
    ticker = yf.Ticker("INR=X")
    data = ticker.history(period="1d", interval="1m")
    return round(data['Close'].iloc[-1], 4)

# ── Signal Logic ───────────────────────────────────────────────
def get_signal(pct, confidence):
    if confidence > 0.005:
        return "⚪ HOLD"
    if pct >= 0.08:
        return "🟢 STRONG BUY"
    elif pct >= 0.03:
        return "🟩 BUY"
    elif pct <= -0.08:
        return "🔴 STRONG SELL"
    elif pct <= -0.03:
        return "🟥 SELL"
    else:
        return "⚪ HOLD"

# ── Prediction Model ───────────────────────────────────────────
def predict():
    df      = get_usd_inr_data()
    current = get_current_price()

    model = Prophet(
        changepoint_prior_scale=0.05,
        daily_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(df)

    results = {"current": current, "predictions": {}}

    for label, freq in HORIZONS.items():
        minutes  = int(freq.replace("min", ""))
        future   = model.make_future_dataframe(periods=1, freq=f"{minutes}min")
        forecast = model.predict(future)

        predicted  = round(forecast['yhat'].iloc[-1], 4)
        upper      = round(forecast['yhat_upper'].iloc[-1], 4)
        lower      = round(forecast['yhat_lower'].iloc[-1], 4)
        change     = round(predicted - current, 4)
        pct        = round((change / current) * 100, 3)
        confidence = round((upper - lower) / predicted, 5)
        signal     = get_signal(pct, confidence)

        results["predictions"][label] = {
            "price": predicted, "change": change, "pct": pct,
            "upper": upper, "lower": lower,
            "confidence": confidence, "signal": signal
        }

    return results

# ── Graph Generator ────────────────────────────────────────────
def generate_graph(data, horizon=None):
    current = data['current']
    preds   = data['predictions']
    now     = datetime.now()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    if horizon:
        vals   = preds[horizon]
        t_end  = now + timedelta(minutes=OFFSET_MINS[horizon])
        times  = [now, t_end]
        prices = [current, vals['price']]
        uppers = [current, vals['upper']]
        lowers = [current, vals['lower']]
        color  = SIGNAL_COLORS.get(vals['signal'], "#90a4ae")

        ax.plot(times, prices, color=color, linewidth=2.5, marker='o', markersize=7)
        ax.fill_between(times, lowers, uppers, color=color, alpha=0.15, label="Confidence Band")

        sign = "+" if vals['change'] >= 0 else ""
        ax.annotate(
            f"{vals['price']}\n({sign}{vals['pct']}%)\n{vals['signal']}",
            xy=(t_end, vals['price']),
            xytext=(10, 10), textcoords='offset points',
            color=color, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2e2e3e', edgecolor=color)
        )
        ax.set_title(f"USD/INR — {horizon} Forecast", color='white', fontsize=14, fontweight='bold')

    else:
        times  = [now] + [now + timedelta(minutes=OFFSET_MINS[h]) for h in preds]
        prices = [current] + [v['price'] for v in preds.values()]

        ax.plot(times, prices, color="#7c83ff", linewidth=2, alpha=0.4, zorder=1)

        for i, (h, vals) in enumerate(preds.items()):
            color = SIGNAL_COLORS.get(vals['signal'], "#90a4ae")
            ax.scatter(times[i+1], vals['price'], color=color, s=80, zorder=5)
            sign = "+" if vals['change'] >= 0 else ""
            ax.annotate(
                f"{h}\n{sign}{vals['pct']}%",
                xy=(times[i+1], vals['price']),
                xytext=(0, 12), textcoords='offset points',
                color='white', fontsize=7, ha='center', rotation=20
            )

        ax.set_title("USD/INR — All Horizons Forecast", color='white', fontsize=14, fontweight='bold')
        patches = [mpatches.Patch(color=c, label=s) for s, c in SIGNAL_COLORS.items()]
        ax.legend(handles=patches, facecolor='#2e2e3e', labelcolor='white', fontsize=8)

    ax.axhline(y=current, color='#ffeb3b', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.tick_params(colors='white')
    ax.xaxis.set_tick_params(rotation=25)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.set_ylabel("USD/INR Rate", color='white')
    ax.set_xlabel("Time", color='white')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    return buf

# ── Keyboard Builders ──────────────────────────────────────────
def horizons_keyboard():
    labels = list(HORIZONS.keys())
    rows = []
    for i in range(0, len(labels), 3):
        row = [
            InlineKeyboardButton(f"🕐 {l}", callback_data=f"horizon:{l}")
            for l in labels[i:i+3]
        ]
        rows.append(row)
    rows.append([InlineKeyboardButton("📊 All Horizons", callback_data="horizon:ALL")])
    return InlineKeyboardMarkup(rows)

def back_keyboard():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔙 Back to Horizons", callback_data="back:choose")
    ]])

# ── Formatters ─────────────────────────────────────────────────
def format_single(data, horizon):
    vals    = data['predictions'][horizon]
    current = data['current']
    sign    = "+" if vals['change'] >= 0 else ""
    arrow   = "📈" if vals['change'] >= 0 else "📉"
    return (
        f"💱 *USD/INR — {horizon}*\n"
        f"📍 Current:    `{current}`\n"
        f"{arrow} Predicted:  `{vals['price']}` ({sign}{vals['pct']}%)\n"
        f"📶 Signal:      {vals['signal']}\n"
        f"📏 Range:       `{vals['lower']}` – `{vals['upper']}`\n"
        f"🎯 Confidence: `{vals['confidence']}`\n\n"
        f"_⚠️ Not financial advice._"
    )

def format_all(data):
    current = data['current']
    lines   = [f"💱 *USD/INR Full Forecast*\n📍 Current: `{current}`\n"]
    for h, vals in data['predictions'].items():
        sign  = "+" if vals['change'] >= 0 else ""
        arrow = "📈" if vals['change'] >= 0 else "📉"
        lines.append(
            f"{arrow} *{h}*: `{vals['price']}` ({sign}{vals['pct']}%)  {vals['signal']}"
        )
    lines.append("\n_⚠️ Not financial advice._")
    return "\n".join(lines)

# ── Command Handlers ───────────────────────────────────────────
async def set_commands(app):
    await app.bot.set_my_commands([
        BotCommand("start",   "👋 Welcome message"),
        BotCommand("predict", "📊 Full forecast all horizons"),
        BotCommand("quick",   "⚡ Short-term signals (10m–1hr)"),
        BotCommand("choose",  "🎯 Pick a specific time horizon"),
        BotCommand("help",    "📖 Help & signal legend"),
    ])

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to USD/INR Forecast Bot!*\n\n"
        "🔮 Powered by Prophet ML model\n\n"
        "📌 *Commands:*\n"
        "/predict — Full forecast (all horizons)\n"
        "/quick   — Short-term signals (10m–1hr)\n"
        "/choose  — Pick a specific time horizon\n"
        "/help    — Signal legend & info\n\n"
        "━━━━━━━━━━━━━━━━━\n"
        "👤 *Created by* [jerin\\_7575](https://www.instagram.com/jerin_7575?igsh=MWVuZ211MHluOXFsNA==)\n"
        "━━━━━━━━━━━━━━━━━",
        parse_mode="Markdown",
        disable_web_page_preview=True
    )

async def choose_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎯 *Select a Time Horizon:*",
        parse_mode="Markdown",
        reply_markup=horizons_keyboard()
    )

async def predict_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Running full forecast...")
    try:
        data  = predict()
        graph = generate_graph(data)
        await update.message.reply_photo(
            photo=graph,
            caption=format_all(data),
            parse_mode="Markdown"
        )
        await msg.delete()
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)}")

async def quick_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⚡ Fetching short-term signals...")
    try:
        data    = predict()
        current = data['current']
        lines   = [f"⚡ *Quick Signals*\n📍 Current: `{current}`\n"]
        for h in ["10 mins", "30 mins", "1 hour"]:
            vals  = data['predictions'][h]
            sign  = "+" if vals['change'] >= 0 else ""
            arrow = "📈" if vals['change'] >= 0 else "📉"
            lines.append(f"{arrow} *{h}*: `{vals['price']}` ({sign}{vals['pct']}%)  {vals['signal']}")
        lines.append("\n_⚠️ Not financial advice._")
        graph = generate_graph(data, horizon="1 hour")
        await update.message.reply_photo(
            photo=graph,
            caption="\n".join(lines),
            parse_mode="Markdown"
        )
        await msg.delete()
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)}")

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Help & Signal Legend*\n\n"
        "🟢 *STRONG BUY*  → +0.08% or more\n"
        "🟩 *BUY*              → +0.03% to +0.08%\n"
        "⚪ *HOLD*             → low move or uncertain\n"
        "🟥 *SELL*              → -0.03% to -0.08%\n"
        "🔴 *STRONG SELL* → -0.08% or more\n\n"
        "📏 *Range* = Prophet confidence interval\n"
        "🎯 *Confidence* = lower is more certain\n\n"
        "_⚠️ This bot is for informational purposes only. Not financial advice._",
        parse_mode="Markdown"
    )

# ── Inline Button Handler ──────────────────────────────────────
async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query    = update.callback_query
    await query.answer()
    data_str = query.data

    if data_str == "back:choose":
        await query.edit_message_text(
            "🎯 *Select a Time Horizon:*",
            parse_mode="Markdown",
            reply_markup=horizons_keyboard()
        )
        return

    if data_str.startswith("horizon:"):
        horizon = data_str.split("horizon:")[1]
        await query.edit_message_text("⏳ Fetching prediction...")
        try:
            data = predict()
            if horizon == "ALL":
                graph   = generate_graph(data)
                caption = format_all(data)
            else:
                graph   = generate_graph(data, horizon=horizon)
                caption = format_single(data, horizon)

            await query.message.reply_photo(
                photo=graph,
                caption=caption,
                parse_mode="Markdown",
                reply_markup=back_keyboard()
            )
            await query.message.delete()
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)}")

# ── Run Bot ────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(set_commands).build()
    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("quick",   quick_cmd))
    app.add_handler(CommandHandler("choose",  choose_cmd))
    app.add_handler(CommandHandler("help",    help_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))

    print("✅ Bot running!")
    app.run_polling()
