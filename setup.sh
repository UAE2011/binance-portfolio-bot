#!/bin/bash
# ============================================================
# Binance Portfolio Bot — One-Command Setup
# Run this script after cloning the repo on your local machine
# ============================================================

set -e

echo "============================================"
echo "  Binance Portfolio Bot — Setup"
echo "============================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "   Install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    echo "   After installing, restart your terminal and run this script again."
    exit 1
fi

# Check if docker-compose or docker compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE="docker compose"
else
    echo "❌ Docker Compose is not installed."
    echo "   It should come with Docker Desktop. Please reinstall Docker Desktop."
    exit 1
fi

echo "✅ Docker found: $(docker --version)"
echo "✅ Compose found: $($COMPOSE version 2>/dev/null || echo 'available')"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "   Copy .env.example to .env and fill in your API keys:"
    echo "   cp .env.example .env"
    exit 1
fi

echo "✅ .env file found"
echo ""

# Create data directory
mkdir -p data logs

# Build and start the bot
echo "🔨 Building Docker image..."
$COMPOSE build

echo ""
echo "🚀 Starting the bot..."
$COMPOSE up -d

echo ""
echo "============================================"
echo "  ✅ Bot is now running!"
echo "============================================"
echo ""
echo "  Useful commands:"
echo "    View logs:     $COMPOSE logs -f"
echo "    Stop bot:      $COMPOSE down"
echo "    Restart bot:   $COMPOSE restart"
echo "    Bot status:    $COMPOSE ps"
echo ""
echo "  Telegram commands:"
echo "    /status  — Portfolio overview"
echo "    /balance — Wallet balance"
echo "    /trades  — Recent trades"
echo "    /ai      — AI advisor status"
echo "    /risk    — Risk metrics"
echo "    /help    — All commands"
echo ""
echo "  The bot will send you a startup message on Telegram."
echo "============================================"
