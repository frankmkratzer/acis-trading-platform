#!/bin/bash

# ACIS Trading Platform Digital Ocean Deployment Script

set -e  # Exit on any error

echo "🚀 ACIS Trading Platform - Digital Ocean Deployment"
echo "=================================================="

# Configuration
DROPLET_NAME="acis-trading-platform"
REGION="nyc3"
SIZE="s-4vcpu-8gb"
IMAGE="ubuntu-22-04-x64"
SSH_KEY_NAME="acis-trading-key"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
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
    
    if ! command -v doctl &> /dev/null; then
        print_error "doctl CLI not found. Please install: https://docs.digitalocean.com/reference/doctl/how-to/install/"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    if ! doctl auth list &> /dev/null; then
        print_error "doctl not authenticated. Run: doctl auth init"
        exit 1
    fi
    
    print_status "Prerequisites check passed ✓"
}

# Create SSH key if it doesn't exist
setup_ssh_key() {
    print_status "Setting up SSH key..."
    
    if ! doctl compute ssh-key list | grep -q "$SSH_KEY_NAME"; then
        if [ ! -f ~/.ssh/acis_trading_rsa ]; then
            print_status "Generating new SSH key..."
            ssh-keygen -t rsa -b 4096 -f ~/.ssh/acis_trading_rsa -N ""
        fi
        
        print_status "Uploading SSH key to Digital Ocean..."
        doctl compute ssh-key import "$SSH_KEY_NAME" --public-key-file ~/.ssh/acis_trading_rsa.pub
    else
        print_status "SSH key already exists ✓"
    fi
}

# Create droplet
create_droplet() {
    print_status "Creating Digital Ocean droplet..."
    
    if doctl compute droplet list | grep -q "$DROPLET_NAME"; then
        print_warning "Droplet $DROPLET_NAME already exists"
        DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep "$DROPLET_NAME" | awk '{print $2}')
    else
        print_status "Creating new droplet: $DROPLET_NAME"
        doctl compute droplet create "$DROPLET_NAME" \
            --region "$REGION" \
            --size "$SIZE" \
            --image "$IMAGE" \
            --ssh-keys "$SSH_KEY_NAME" \
            --enable-monitoring \
            --enable-backups \
            --wait
        
        DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep "$DROPLET_NAME" | awk '{print $2}')
    fi
    
    print_status "Droplet IP: $DROPLET_IP"
    
    # Wait for SSH to be available
    print_status "Waiting for SSH to be available..."
    while ! nc -z "$DROPLET_IP" 22; do
        sleep 5
    done
    
    sleep 30  # Additional wait for system initialization
}

# Setup server
setup_server() {
    print_status "Setting up server environment..."
    
    ssh -i ~/.ssh/acis_trading_rsa -o StrictHostKeyChecking=no root@"$DROPLET_IP" << 'EOF'
        # Update system
        apt-get update && apt-get upgrade -y
        
        # Install Docker
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        
        # Install Docker Compose
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
        
        # Create application directory
        mkdir -p /opt/acis-trading
        mkdir -p /var/log/acis-trading
        
        # Install monitoring tools
        apt-get install -y htop iotop net-tools
        
        # Setup log rotation
        cat > /etc/logrotate.d/acis-trading << 'LOGROTATE'
/var/log/acis-trading/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
LOGROTATE
        
        # Enable Docker service
        systemctl enable docker
        systemctl start docker
        
        echo "✓ Server setup complete"
EOF
}

# Deploy application
deploy_application() {
    print_status "Deploying ACIS Trading Platform..."
    
    # Copy application files
    print_status "Copying application files..."
    scp -i ~/.ssh/acis_trading_rsa -r ./* root@"$DROPLET_IP":/opt/acis-trading/
    
    # Setup environment file
    ssh -i ~/.ssh/acis_trading_rsa root@"$DROPLET_IP" << 'EOF'
        cd /opt/acis-trading
        
        # Create environment file template
        cat > .env.template << 'ENVFILE'
# ACIS Trading Platform Environment Configuration
POSTGRES_URL=postgresql://acis_user:YOUR_PASSWORD@postgres:5432/acis_trading
POSTGRES_PASSWORD=YOUR_SECURE_PASSWORD
ENVIRONMENT=production

# Optional: API Keys
ALPHA_VANTAGE_API_KEY=your_api_key_here
FINANCIAL_MODELING_PREP_KEY=your_key_here

# Monitoring
GRAFANA_PASSWORD=your_grafana_password

# Alerts (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL=alerts@yourdomain.com
ENVFILE

        echo "⚠️  Please edit .env file with your actual configuration:"
        echo "   nano .env.template"
        echo "   mv .env.template .env"
        
EOF

    print_status "Application files deployed ✓"
}

# Start services
start_services() {
    print_status "Starting ACIS Trading Platform services..."
    
    ssh -i ~/.ssh/acis_trading_rsa root@"$DROPLET_IP" << 'EOF'
        cd /opt/acis-trading
        
        if [ ! -f .env ]; then
            echo "❌ .env file not found! Please create it first."
            echo "   cp .env.template .env"
            echo "   nano .env"
            exit 1
        fi
        
        # Build and start services
        docker-compose build
        docker-compose up -d
        
        # Show running services
        docker-compose ps
        
        echo "✅ ACIS Trading Platform is running!"
        echo ""
        echo "📊 Service URLs:"
        echo "   - Logs: docker-compose logs -f acis-trading"
        echo "   - Status: docker-compose ps"
        echo "   - Stop: docker-compose down"
        echo ""
        echo "🔧 Management Commands:"
        echo "   - Manual execution: docker-compose exec acis-trading python smart_scheduler.py test"
        echo "   - Database access: docker-compose exec postgres psql -U acis_user acis_trading"
        echo ""
EOF
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring (optional)..."
    
    ssh -i ~/.ssh/acis_trading_rsa root@"$DROPLET_IP" << 'EOF'
        cd /opt/acis-trading
        
        # Start monitoring stack
        docker-compose --profile monitoring up -d
        
        echo "📈 Monitoring URLs:"
        echo "   - Grafana: http://DROPLET_IP:3000 (admin/admin)"
        echo "   - Prometheus: http://DROPLET_IP:9090"
EOF
}

# Setup SSL (optional)
setup_ssl() {
    print_status "Setting up SSL with Let's Encrypt (optional)..."
    
    read -p "Enter your domain name (or press Enter to skip): " DOMAIN
    
    if [ -n "$DOMAIN" ]; then
        ssh -i ~/.ssh/acis_trading_rsa root@"$DROPLET_IP" << EOF
            # Install Nginx and Certbot
            apt-get install -y nginx certbot python3-certbot-nginx
            
            # Configure Nginx
            cat > /etc/nginx/sites-available/acis-trading << 'NGINX'
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
NGINX

            ln -s /etc/nginx/sites-available/acis-trading /etc/nginx/sites-enabled/
            nginx -t && systemctl reload nginx
            
            # Get SSL certificate
            certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN
            
            echo "✅ SSL setup complete for $DOMAIN"
EOF
    fi
}

# Main deployment flow
main() {
    print_status "Starting ACIS Trading Platform deployment..."
    
    check_prerequisites
    setup_ssh_key
    create_droplet
    setup_server
    deploy_application
    
    print_status "🎉 Deployment completed successfully!"
    print_status "💡 Next steps:"
    echo "   1. SSH to your server: ssh -i ~/.ssh/acis_trading_rsa root@$DROPLET_IP"
    echo "   2. Configure environment: cd /opt/acis-trading && nano .env"
    echo "   3. Start services: docker-compose up -d"
    echo "   4. View logs: docker-compose logs -f"
    echo ""
    print_status "📋 Server Details:"
    echo "   - IP Address: $DROPLET_IP"
    echo "   - SSH Command: ssh -i ~/.ssh/acis_trading_rsa root@$DROPLET_IP"
    echo "   - Monthly Cost: ~$48 (4 vCPU, 8GB RAM)"
    echo ""
    
    # Ask about optional features
    read -p "Setup monitoring stack? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_monitoring
    fi
    
    read -p "Setup SSL certificate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_ssl
    fi
    
    print_status "🚀 ACIS Trading Platform is ready for production!"
}

# Run main function
main "$@"