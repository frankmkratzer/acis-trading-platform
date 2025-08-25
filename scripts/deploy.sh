#!/bin/bash
# deploy.sh - Alpha Centauri Investment Strategies (ACIS) Deployment Script

set -e

echo "🚀 Starting ACIS (Alpha Centauri Investment Strategies) Platform Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DROPLET_SIZE="s-4vcpu-8gb"  # 8GB RAM, 4 CPUs
REGION="nyc1"               # New York datacenter (close to your existing DB)
DOMAIN="acis.your-domain.com"    # Update with your actual domain

print_status() {
    echo -e "${BLUE}[ACIS]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if doctl is installed
    if ! command -v doctl &> /dev/null; then
        print_error "DigitalOcean CLI (doctl) is not installed"
        echo "Install it: https://docs.digitalocean.com/reference/doctl/how-to/install/"
        exit 1
    fi

    # Check if docker is available locally for building
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        echo "Install it: https://docs.docker.com/get-docker/"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Create Digital Ocean droplet
create_droplet() {
    print_status "Creating ACIS application server..."

    # Create droplet
    DROPLET_ID=$(doctl compute droplet create acis-platform \
        --size $DROPLET_SIZE \
        --region $REGION \
        --image ubuntu-22-04-x64 \
        --ssh-keys $(doctl compute ssh-key list --format ID --no-header | head -1) \
        --enable-monitoring \
        --enable-private-networking \
        --format ID \
        --no-header)

    print_success "ACIS droplet created with ID: $DROPLET_ID"

    # Wait for droplet to be ready
    print_status "Waiting for server to initialize..."
    sleep 60

    # Get droplet IP
    DROPLET_IP=$(doctl compute droplet get $DROPLET_ID --format PublicIPv4 --no-header)
    print_success "ACIS Platform IP: $DROPLET_IP"

    echo "DROPLET_IP=$DROPLET_IP" > .env.deployment
}

# Main deployment function
main() {
    print_status "Starting ACIS (Alpha Centauri Investment Strategies) deployment..."

    check_prerequisites

    # Since your API keys are already configured, we can proceed
    print_status "Using your existing credentials:"
    echo "  - PostgreSQL Database: DigitalOcean Managed Database"
    echo "  - Alpha Vantage API: CBWWW6YNG62LUZ60"
    echo "  - Schwab Trading API: yr8syNnNIAGrFgaGbiNOGk2ahki8Lh5n"
    echo ""

    read -p "Ready to deploy ACIS platform? (y/N): " confirm

    if [[ $confirm != [yY] ]]; then
        print_error "Deployment cancelled"
        exit 1
    fi

    create_droplet

    print_success "🎉 ACIS Platform deployment initiated!"
    print_status "Your quantitative trading platform setup is ready! 🚀"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi