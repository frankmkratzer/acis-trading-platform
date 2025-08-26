# Simple Digital Ocean Setup for ACIS Trading Platform

## Quick Setup (5 Steps)

### 1. Create DigitalOcean Droplet
```bash
# Option A: Through Web Interface (Recommended)
1. Go to cloud.digitalocean.com
2. Click "Create" → "Droplets"
3. Choose: Ubuntu 22.04 LTS
4. Size: Basic plan, 4 vCPUs, 8GB RAM ($48/month)
5. Add your SSH key
6. Create droplet

# Option B: CLI Method
doctl compute droplet create acis-trading \
  --size s-4vcpu-8gb \
  --image ubuntu-22-04-x64 \
  --region nyc1 \
  --ssh-keys YOUR_SSH_KEY_ID
```

### 2. Connect and Setup Server
```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

### 3. Upload Your Code
```bash
# From your local machine, upload the ACIS platform
scp -r acis-trading-platform root@YOUR_DROPLET_IP:/opt/

# Or clone from repository
ssh root@YOUR_DROPLET_IP
cd /opt
git clone https://github.com/your-username/acis-trading-platform.git
```

### 4. Configure Environment
```bash
# Create production environment file
cd /opt/acis-trading-platform
cp .env.example .env

# Edit with your settings
nano .env
```

**Required .env Settings:**
```env
POSTGRES_PASSWORD=your_secure_password_here
SECRET_KEY=your_flask_secret_key_here
SMTP_HOST=smtp.gmail.com
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=your_email_password
ALERT_EMAIL=admin@yourdomain.com

# Broker API Keys (optional for now)
SCHWAB_CLIENT_ID=your_schwab_id
SCHWAB_CLIENT_SECRET=your_schwab_secret
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
```

### 5. Start the Platform
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d --build

# Check status
docker ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f admin_app
```

## Access Your Platform

- **Admin Interface**: `http://YOUR_DROPLET_IP:5001`
- **Client Dashboard**: `http://YOUR_DROPLET_IP:5002`
- **Default Login**: admin / admin123

## Quick Commands

```bash
# Restart services
docker-compose -f docker-compose.prod.yml restart

# Update code
git pull
docker-compose -f docker-compose.prod.yml up -d --build

# Backup database
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U acis_user acis_trading > backup.sql

# View system resources
htop
docker stats
```

## Domain Setup (Optional)

1. Point your domain to droplet IP
2. Install SSL certificate:
```bash
apt install certbot nginx
certbot --nginx -d yourdomain.com
```

## Cost Estimate

- **Droplet**: $48/month (4vCPU, 8GB RAM)
- **Backups**: $4.80/month (10% of droplet cost)
- **Load Balancer**: $12/month (if needed)
- **Total**: ~$65/month

## Monitoring

The platform includes built-in monitoring:
- System health dashboard
- Email alerts for issues  
- Automatic backups
- Performance metrics
- Trading system monitoring

## Support

If issues arise:
1. Check logs: `docker-compose logs service_name`
2. Restart services: `docker-compose restart`
3. Monitor resources: `htop` and `docker stats`