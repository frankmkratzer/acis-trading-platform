# ACIS Trading Platform - Comprehensive Admin System

## Overview

The ACIS Trading Platform Admin System is a full-featured web application for managing all aspects of your quantitative trading operations. This comprehensive system provides:

- **User Management** - Role-based access control with multiple permission levels
- **Portfolio Management** - Real-time portfolio monitoring and rebalancing
- **Strategy Management** - Configuration and monitoring of all 12 trading strategies
- **Trade Management** - Order execution, monitoring, and history tracking
- **Client Management** - Client onboarding, reporting, and communication
- **Broker Integration** - Multi-broker support including Schwab, Alpaca, Interactive Brokers
- **System Monitoring** - Real-time performance monitoring and alerting
- **Risk Management** - Comprehensive risk controls and limits
- **Reporting** - Automated report generation and analytics

## System Architecture

### Core Components

1. **Admin Application** (`admin_app.py`) - Main Flask web application
2. **User Authentication** - Role-based access control system
3. **Database Layer** - PostgreSQL with SQLAlchemy ORM
4. **Broker Integration** - Multi-broker trading connectivity
5. **Monitoring System** - Real-time system and trading metrics
6. **Client Dashboard** - Professional client reporting interface
7. **API Layer** - RESTful APIs for all operations

### Technology Stack

- **Backend**: Flask, SQLAlchemy, PostgreSQL
- **Frontend**: Bootstrap 5, Chart.js, jQuery, DataTables
- **Trading**: Multi-broker API integration (Schwab, Alpaca, IB)
- **Monitoring**: psutil, automated alerting, email notifications
- **Deployment**: Docker, Digital Ocean, nginx

## Quick Start

### Prerequisites

1. **Python 3.11+** with pip
2. **PostgreSQL 12+** database
3. **Environment Variables** configured
4. **Broker API Keys** (optional for live trading)

### Installation

1. **Clone the repository and install dependencies:**
```bash
cd acis-trading-platform
pip install -r requirements.txt
```

2. **Set up environment variables (.env file):**
```env
# Database
POSTGRES_URL=postgresql://username:password@localhost:5432/acis_trading

# Admin System
SECRET_KEY=your-secret-key-here
ADMIN_PORT=5001

# Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=admin@yourdomain.com

# Broker APIs (Optional)
SCHWAB_CLIENT_ID=your-schwab-client-id
SCHWAB_CLIENT_SECRET=your-schwab-client-secret
SCHWAB_REDIRECT_URI=https://localhost:8080/callback

ALPACA_API_KEY=your-alpaca-key
ALPACA_SECRET_KEY=your-alpaca-secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Interactive Brokers
IB_GATEWAY_HOST=127.0.0.1
IB_GATEWAY_PORT=7497
IB_CLIENT_ID=1
```

3. **Initialize the database:**
```bash
python admin_app.py
```

4. **Start the admin system:**
```bash
python admin_app.py
```

5. **Access the admin interface:**
   - URL: `http://localhost:5001`
   - Default Login: `admin` / `admin123`

## User Roles and Permissions

### Administrator (`admin`)
- Full system access
- User management
- System configuration
- All trading operations
- All reports and analytics

### Portfolio Manager (`portfolio_manager`)
- Portfolio monitoring and management
- Strategy configuration
- Trade execution
- Client reporting
- Risk monitoring

### Trader (`trader`)
- Order execution
- Position monitoring
- Trade history
- Basic portfolio analytics

### Research Analyst (`analyst`)
- Strategy research and analysis
- Performance analytics
- Market data access
- Report generation

### Client Service (`client_service`)
- Client management
- Report generation
- Portfolio viewing
- Client communication

### Risk Manager (`risk_manager`)
- Risk monitoring and controls
- Alert management
- System oversight
- Compliance reporting

### Read Only (`readonly`)
- View-only access to portfolios
- Trade history viewing
- Basic reporting

## Key Features

### 1. Dashboard Overview

The main dashboard provides:
- **Real-time portfolio metrics** - Total value, positions, performance
- **Performance charts** - 30-day, 90-day, 1-year views
- **Recent trading activity** - Latest orders and executions  
- **System alerts** - Critical notifications and warnings
- **Quick actions** - One-click strategy runs and rebalancing

### 2. Portfolio Management

**Features:**
- Multi-strategy portfolio tracking
- Real-time position monitoring
- Performance attribution analysis
- Sector and market cap allocation
- Risk metrics and controls

**Operations:**
- Manual rebalancing
- Position size adjustments
- Strategy allocation changes
- Risk limit modifications

### 3. Strategy Management

**12 Strategies Supported:**
- Small Cap: Value, Growth, Momentum, Dividend
- Mid Cap: Value, Growth, Momentum, Dividend  
- Large Cap: Value, Growth, Momentum, Dividend

**Management Functions:**
- Strategy configuration
- Performance monitoring
- Parameter adjustments
- Execution scheduling
- Backtesting results

### 4. Trade Management

**Order Management:**
- Order creation and submission
- Real-time execution monitoring
- Order modification and cancellation
- Fill tracking and reporting

**Trade Analytics:**
- Execution quality analysis
- Slippage and market impact
- Transaction cost analysis
- Trade attribution

### 5. Broker Integration

**Supported Brokers:**

#### Charles Schwab
- OAuth 2.0 authentication
- Live and paper trading
- Real-time market data
- Order execution and monitoring
- Position and account data

#### Alpaca Markets
- API key authentication
- Commission-free trading
- Paper trading environment
- Real-time streaming data

#### Interactive Brokers
- TWS/Gateway integration
- Professional trading tools
- Global market access
- Advanced order types

**Configuration:**
- Broker credentials management
- Trading mode selection (paper/live)
- Risk limits per broker
- Automated failover

### 6. Client Management

**Client Features:**
- Client onboarding workflow
- Investment preferences tracking
- Performance reporting
- Communication history

**Reporting:**
- Professional PDF reports
- Interactive web dashboards
- Custom analytics
- Automated delivery

### 7. System Monitoring

**Monitoring Components:**
- System performance (CPU, memory, disk)
- Database connectivity and performance
- Trading system health
- External API connectivity
- Network and security metrics

**Alerting:**
- Real-time alert generation
- Email/SMS notifications
- Alert acknowledgment
- Escalation procedures
- Historical alert tracking

**Metrics Tracked:**
- Portfolio performance
- Risk exposures
- Trade execution quality
- System reliability
- User activity

### 8. Risk Management

**Risk Controls:**
- Position size limits
- Sector concentration limits
- Daily loss limits
- Drawdown monitoring
- Leverage restrictions

**Monitoring:**
- Real-time risk calculations
- Violation alerts
- Historical risk reporting
- Compliance tracking

## API Endpoints

### Authentication
- `POST /login` - User login
- `GET /logout` - User logout

### Portfolio Management
- `GET /api/portfolios` - List all portfolios
- `GET /api/portfolios/{strategy}` - Get portfolio details
- `POST /api/portfolios/{strategy}/rebalance` - Rebalance portfolio

### Trading
- `POST /api/orders` - Submit order
- `GET /api/orders` - List orders
- `GET /api/orders/{id}` - Order details
- `DELETE /api/orders/{id}` - Cancel order

### System Monitoring
- `GET /api/system/status` - System status
- `GET /api/system/alerts` - Active alerts
- `GET /api/system/metrics` - Performance metrics

### Reporting
- `GET /api/reports/portfolio/{strategy}` - Portfolio report
- `POST /api/reports/generate` - Generate custom report

## Deployment

### Development Environment

1. **Start monitoring system:**
```bash
python system_monitoring.py &
```

2. **Start admin application:**
```bash
python admin_app.py
```

3. **Start client dashboard (if needed):**
```bash
python client_dashboard_system.py
```

### Production Deployment

1. **Use Docker Compose:**
```bash
docker-compose up -d
```

2. **Or deploy to Digital Ocean:**
```bash
./deploy.sh
```

3. **Configure nginx proxy:**
```nginx
server {
    listen 80;
    server_name admin.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Environment Configuration

**Production Settings:**
```env
FLASK_DEBUG=False
ADMIN_PORT=5001
SECRET_KEY=production-secret-key-change-this
POSTGRES_URL=postgresql://username:password@prod-db:5432/acis_trading
```

**Security Settings:**
- Change default admin password
- Enable HTTPS/SSL
- Configure firewall rules
- Set up database backups
- Enable audit logging

## Security Features

### Authentication & Authorization
- Password hashing with werkzeug
- Session management with Flask-Login
- CSRF protection with Flask-WTF
- Role-based access control
- Failed login attempt tracking

### Data Protection
- Database connection encryption
- Secure API key storage
- Audit logging for all actions
- Data backup and recovery
- Input validation and sanitization

### Network Security
- HTTPS enforcement
- CORS protection
- Rate limiting
- IP whitelisting
- VPN access requirements

## Monitoring and Alerts

### System Alerts

**Critical Alerts:**
- Database connectivity failure
- Trading system errors
- Risk limit violations
- Security incidents

**Warning Alerts:**
- High system resource usage
- Trading performance degradation
- Unusual trading patterns
- Configuration changes

### Performance Monitoring

**Key Metrics:**
- Response times
- Database performance
- Order execution speed
- System reliability
- User activity patterns

## Troubleshooting

### Common Issues

**Database Connection Issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection string
python -c "from sqlalchemy import create_engine; engine = create_engine('your-postgres-url'); print('Connection OK')"
```

**Authentication Problems:**
- Verify user exists in admin_users table
- Check password hash generation
- Confirm role assignments
- Review session configuration

**Trading System Issues:**
- Verify broker API credentials
- Check network connectivity
- Review trading permissions
- Examine order execution logs

**Performance Issues:**
- Monitor system resources
- Check database query performance
- Review application logs
- Analyze network latency

### Log Files

**Application Logs:**
- Admin system: Flask application logs
- Trading system: Order execution logs
- Monitoring: System health logs
- Database: PostgreSQL logs

**Log Locations:**
```
/var/log/acis-trading/admin.log
/var/log/acis-trading/trading.log
/var/log/acis-trading/monitoring.log
/var/log/acis-trading/database.log
```

## Support and Maintenance

### Regular Maintenance

**Daily:**
- Review system alerts
- Check trading performance
- Monitor resource usage
- Verify backup completion

**Weekly:**
- Review user activity
- Update strategy parameters
- Analyze performance reports
- Security audit review

**Monthly:**
- Database optimization
- System updates
- Performance tuning
- Capacity planning

### Backup and Recovery

**Database Backups:**
```bash
# Daily backup
pg_dump acis_trading > backup_$(date +%Y%m%d).sql

# Restore from backup
psql acis_trading < backup_20240826.sql
```

**Configuration Backups:**
- Environment files
- Strategy configurations
- User settings
- System preferences

## Advanced Features

### Custom Strategy Development

**Adding New Strategies:**
1. Define strategy logic in Python
2. Configure database tables
3. Add admin interface forms
4. Implement backtesting
5. Enable live trading

### API Integration

**Custom API Development:**
- RESTful API design
- Authentication integration
- Rate limiting
- Documentation generation
- Testing frameworks

### Reporting Customization

**Custom Reports:**
- Template customization
- Data source integration
- Automated scheduling
- Delivery mechanisms
- Performance optimization

## Conclusion

The ACIS Trading Platform Admin System provides comprehensive management capabilities for quantitative trading operations. With its professional interface, robust security, and extensive monitoring capabilities, it enables efficient operation of sophisticated trading strategies across multiple brokers and client accounts.

For additional support or customization needs, refer to the system documentation or contact the development team.

---

**System Requirements:**
- Python 3.11+
- PostgreSQL 12+
- 4GB RAM minimum
- 50GB storage minimum
- SSL certificate for production
- SMTP server for alerts

**Recommended Hardware:**
- 8GB+ RAM for production
- SSD storage for database
- Redundant network connections
- UPS power backup
- Monitoring and alerting infrastructure