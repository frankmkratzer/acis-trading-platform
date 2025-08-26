# ACIS Trading Platform - Production Deployment Checklist

## Pre-Deployment Requirements

### 1. Environment Configuration
- [ ] Create `.env.prod` file with production environment variables
- [ ] Set secure passwords for all services (PostgreSQL, Redis, Admin users)
- [ ] Configure SMTP settings for email alerts
- [ ] Set up broker API credentials (Schwab, Alpaca, Interactive Brokers)
- [ ] Generate secure secret keys for Flask applications

### 2. Domain and DNS Setup
- [ ] Register domain names (admin.acis-trading.com, client.acis-trading.com)
- [ ] Configure DNS records pointing to server IP
- [ ] Plan SSL certificate acquisition (Let's Encrypt recommended)

### 3. DigitalOcean Setup
- [ ] Install DigitalOcean CLI (`doctl`)
- [ ] Authenticate with DigitalOcean: `doctl auth init`
- [ ] Generate SSH keys for server access
- [ ] Plan droplet size (recommended: s-4vcpu-8gb or larger)

### 4. Security Preparation
- [ ] Review and update default passwords
- [ ] Configure firewall rules (ports 22, 80, 443)
- [ ] Set up VPN access if required
- [ ] Plan backup encryption keys

## Production Environment Variables (.env.prod)

```env
# Database Configuration
POSTGRES_PASSWORD=your-secure-password-here
POSTGRES_URL=postgresql://acis_user:your-secure-password-here@postgres:5432/acis_trading

# Application Security
SECRET_KEY=your-flask-secret-key-here
CLIENT_SECRET_KEY=your-client-secret-key-here

# Email Configuration for Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=admin@yourdomain.com

# Broker API Configuration
# Charles Schwab
SCHWAB_CLIENT_ID=your-schwab-client-id
SCHWAB_CLIENT_SECRET=your-schwab-client-secret
SCHWAB_REDIRECT_URI=https://admin.acis-trading.com/callback

# Alpaca Markets
ALPACA_API_KEY=your-alpaca-api-key
ALPACA_SECRET_KEY=your-alpaca-secret-key
ALPACA_BASE_URL=https://api.alpaca.markets  # Live trading
# ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Interactive Brokers
IB_GATEWAY_HOST=127.0.0.1
IB_GATEWAY_PORT=7497
IB_CLIENT_ID=1

# Backup Configuration
BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
```

## Deployment Steps

### 1. Server Preparation
```bash
# Clone repository
git clone https://github.com/your-org/acis-trading-platform.git
cd acis-trading-platform

# Make deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### 2. Docker Deployment
```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d --build

# Verify services
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs -f admin_app
```

### 3. Database Initialization
```bash
# Access admin container
docker-compose -f docker-compose.prod.yml exec admin_app bash

# Initialize database tables
python -c "
from admin_app import init_admin_tables
init_admin_tables()
print('Database initialized successfully')
"
```

### 4. SSL Certificate Setup
```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Generate SSL certificates
sudo certbot --nginx -d admin.acis-trading.com -d client.acis-trading.com

# Test auto-renewal
sudo certbot renew --dry-run
```

## Post-Deployment Verification

### 1. Service Health Checks
- [ ] Admin interface accessible at https://admin.acis-trading.com
- [ ] Client dashboard accessible at https://client.acis-trading.com
- [ ] All Docker containers running: `docker ps`
- [ ] Database connectivity working
- [ ] Redis cache operational
- [ ] Monitoring system active

### 2. Security Verification
- [ ] HTTPS enforced on all domains
- [ ] Default passwords changed
- [ ] Firewall rules active
- [ ] SSH key authentication working
- [ ] Rate limiting functional

### 3. Trading System Verification
- [ ] Broker API connections established
- [ ] Paper trading functional
- [ ] Strategy execution working
- [ ] Order management operational
- [ ] Risk controls active

### 4. Monitoring and Alerts
- [ ] System monitoring active
- [ ] Email alerts configured
- [ ] Log rotation setup
- [ ] Backup system operational
- [ ] Performance metrics available

## Production Monitoring

### 1. System Metrics
```bash
# Monitor system resources
htop
docker stats

# Check service logs
docker-compose -f docker-compose.prod.yml logs --tail=100 -f admin_app
docker-compose -f docker-compose.prod.yml logs --tail=100 -f monitoring
```

### 2. Database Monitoring
```bash
# PostgreSQL performance
docker-compose -f docker-compose.prod.yml exec postgres psql -U acis_user -d acis_trading -c "
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables 
ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC;
"
```

### 3. Application Health
- Admin interface response time
- API endpoint performance
- Trading system latency
- Database query performance
- Memory and CPU usage

## Maintenance Procedures

### 1. Regular Maintenance
- [ ] **Daily**: Review system alerts and logs
- [ ] **Weekly**: Check backup integrity
- [ ] **Monthly**: Update security patches
- [ ] **Quarterly**: Performance optimization review

### 2. Backup and Recovery
```bash
# Manual backup
docker-compose -f docker-compose.prod.yml exec backup python backup_system.py

# Restore from backup
docker-compose -f docker-compose.prod.yml exec postgres psql -U acis_user -d acis_trading < backup.sql
```

### 3. System Updates
```bash
# Update containers
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --remove-orphans

# Update system packages
apt-get update && apt-get upgrade -y
```

## Emergency Procedures

### 1. Service Recovery
```bash
# Restart all services
docker-compose -f docker-compose.prod.yml restart

# Restart specific service
docker-compose -f docker-compose.prod.yml restart admin_app

# Check service status
docker-compose -f docker-compose.prod.yml ps
```

### 2. Database Recovery
```bash
# Check database status
docker-compose -f docker-compose.prod.yml exec postgres pg_isready

# Restore from backup
docker-compose -f docker-compose.prod.yml exec postgres psql -U acis_user -d acis_trading < latest_backup.sql
```

### 3. Emergency Contacts
- System Administrator: admin@yourdomain.com
- DigitalOcean Support: Available 24/7
- Broker Support Contacts (Schwab, Alpaca, IB)

## Performance Optimization

### 1. Database Optimization
- Regular VACUUM and ANALYZE
- Index optimization
- Query performance monitoring
- Connection pooling configuration

### 2. Application Optimization
- Redis caching strategy
- API response optimization
- Static file caching
- CDN implementation

### 3. Infrastructure Scaling
- Droplet sizing recommendations
- Load balancer configuration
- Database read replicas
- Microservice architecture

## Security Best Practices

### 1. Access Control
- Regular password rotation
- SSH key management
- VPN access requirements
- Multi-factor authentication

### 2. Data Protection
- Database encryption at rest
- SSL/TLS for all communications
- API key rotation
- Audit logging

### 3. Network Security
- Firewall configuration
- DDoS protection
- Rate limiting
- Intrusion detection

## Troubleshooting Guide

### Common Issues

1. **Container Won't Start**
   - Check logs: `docker-compose logs service_name`
   - Verify environment variables
   - Check port conflicts

2. **Database Connection Failed**
   - Verify PostgreSQL container is running
   - Check connection string format
   - Verify user permissions

3. **SSL Certificate Issues**
   - Check certificate expiry
   - Verify domain configuration
   - Renew certificates if needed

4. **High Memory Usage**
   - Monitor container memory usage
   - Check for memory leaks
   - Optimize database queries

5. **Slow Response Times**
   - Enable query logging
   - Check database indexes
   - Monitor network latency

## Success Criteria

The deployment is considered successful when:
- [ ] All services are running and healthy
- [ ] Admin interface is accessible and functional
- [ ] Client dashboard is operational
- [ ] Trading system can execute paper trades
- [ ] All broker integrations are connected
- [ ] Monitoring and alerting systems are active
- [ ] Backups are running automatically
- [ ] Security measures are in place and verified

## Support and Documentation

- **System Documentation**: `/app/docs/`
- **API Documentation**: `https://admin.acis-trading.com/docs`
- **Log Files**: `/app/logs/`
- **Backup Files**: `/app/backups/`
- **Support Email**: support@acis-trading.com

---

**Important Notes:**
1. Never deploy to production without testing in a staging environment
2. Always backup before making changes
3. Monitor system performance closely after deployment
4. Keep emergency contact information readily available
5. Document all configuration changes and customizations