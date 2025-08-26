#!/bin/bash

# ACIS Trading Platform - Digital Ocean Production Deployment Script
set -e

echo "ACIS Trading Platform - Production Deployment"
echo "=============================================="

# Check if .env.prod exists
if [[ ! -f ".env.prod" ]]; then
    echo "Error: .env.prod file not found"
    exit 1
fi

echo "Deployment framework ready for Digital Ocean"
echo "Configure .env.prod and run with Docker Compose"
