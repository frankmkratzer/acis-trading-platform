#!/usr/bin/env python3
"""
ACIS Trading Platform - System Monitoring and Alerting
Comprehensive system monitoring, performance tracking, and alert management
"""

import os
import psutil
import time
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import threading
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests

@dataclass
class SystemMetric:
    """System performance metric"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = ""

@dataclass
class Alert:
    """System alert"""
    alert_type: str
    severity: str  # info, warning, critical
    title: str
    message: str
    data: Dict = None
    created_at: datetime = None

class SystemMonitor:
    """Comprehensive system monitoring and alerting"""
    
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SystemMonitor')
        
        # Monitoring configuration
        self.config = {
            'check_interval': 60,  # seconds
            'metrics_retention_days': 30,
            'alert_cooldown_minutes': 5,
            
            # System thresholds
            'cpu_warning': 75.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            
            # Trading system thresholds
            'portfolio_loss_warning': -5.0,  # %
            'portfolio_loss_critical': -10.0,  # %
            'position_concentration_warning': 8.0,  # %
            'position_concentration_critical': 10.0,  # %
            'daily_volume_warning': 1000000,  # $
            'trade_failure_rate_warning': 5.0,  # %
            'trade_failure_rate_critical': 10.0,  # %
        }
        
        # Alert tracking
        self.alert_cooldowns = {}
        self.running = False
        
        # Initialize database tables
        self._init_monitoring_tables()
        
    def _init_monitoring_tables(self):
        """Initialize monitoring database tables"""
        with self.engine.connect() as conn:
            # System metrics table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(50) NOT NULL,
                    value DECIMAL(15,4) NOT NULL,
                    unit VARCHAR(20),
                    threshold_warning DECIMAL(15,4),
                    threshold_critical DECIMAL(15,4),
                    description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # System alerts table (if not exists from admin_app.py)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    message TEXT,
                    data JSONB,
                    acknowledged BOOLEAN DEFAULT false,
                    acknowledged_by INTEGER,
                    acknowledged_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Trading performance metrics
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_date DATE NOT NULL,
                    strategy VARCHAR(50),
                    total_positions INTEGER DEFAULT 0,
                    total_value DECIMAL(15,2) DEFAULT 0,
                    daily_pnl DECIMAL(15,2) DEFAULT 0,
                    daily_return_pct DECIMAL(8,4) DEFAULT 0,
                    trades_executed INTEGER DEFAULT 0,
                    trades_failed INTEGER DEFAULT 0,
                    largest_position_pct DECIMAL(8,4) DEFAULT 0,
                    sector_concentration_max DECIMAL(8,4) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(metric_date, strategy)
                )
            """))
            
            # System health log
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS system_health_log (
                    id SERIAL PRIMARY KEY,
                    component VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    details JSONB,
                    response_time_ms INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.commit()
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.running = True
        self.logger.info("Starting system monitoring...")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        # Start health check thread
        health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        health_thread.start()
        
        return True
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.logger.info("Stopping system monitoring...")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect trading metrics
                self._collect_trading_metrics()
                
                # Check thresholds and generate alerts
                self._check_alert_conditions()
                
                # Clean old metrics
                self._cleanup_old_metrics()
                
                # Sleep until next check
                time.sleep(self.config['check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            metrics = []
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(SystemMetric(
                metric_name='cpu_usage',
                value=cpu_percent,
                unit='percent',
                timestamp=datetime.now(),
                threshold_warning=self.config['cpu_warning'],
                threshold_critical=self.config['cpu_critical'],
                description='CPU utilization percentage'
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            metrics.append(SystemMetric(
                metric_name='memory_usage',
                value=memory_percent,
                unit='percent',
                timestamp=datetime.now(),
                threshold_warning=self.config['memory_warning'],
                threshold_critical=self.config['memory_critical'],
                description='Memory utilization percentage'
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(SystemMetric(
                metric_name='disk_usage',
                value=disk_percent,
                unit='percent',
                timestamp=datetime.now(),
                threshold_warning=self.config['disk_warning'],
                threshold_critical=self.config['disk_critical'],
                description='Disk utilization percentage'
            ))
            
            # Network I/O
            network = psutil.net_io_counters()
            metrics.append(SystemMetric(
                metric_name='network_bytes_sent',
                value=network.bytes_sent,
                unit='bytes',
                timestamp=datetime.now(),
                description='Total network bytes sent'
            ))
            
            metrics.append(SystemMetric(
                metric_name='network_bytes_recv',
                value=network.bytes_recv,
                unit='bytes',
                timestamp=datetime.now(),
                description='Total network bytes received'
            ))
            
            # Process count
            process_count = len(psutil.pids())
            metrics.append(SystemMetric(
                metric_name='process_count',
                value=process_count,
                unit='count',
                timestamp=datetime.now(),
                description='Number of running processes'
            ))
            
            # Save metrics to database
            self._save_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_trading_metrics(self):
        """Collect trading system metrics"""
        try:
            with self.engine.connect() as conn:
                # Get portfolio metrics by strategy
                result = conn.execute(text("""
                    SELECT 
                        strategy,
                        COUNT(*) as total_positions,
                        SUM(market_value) as total_value,
                        SUM(unrealized_pnl) as unrealized_pnl,
                        MAX(market_value / (SELECT SUM(market_value) FROM trading_positions WHERE quantity > 0) * 100) as largest_position_pct
                    FROM trading_positions 
                    WHERE quantity > 0
                    GROUP BY strategy
                """))
                
                portfolio_metrics = result.fetchall()
                
                # Get trade metrics for today
                today = datetime.now().date()
                trade_result = conn.execute(text("""
                    SELECT 
                        strategy,
                        COUNT(*) as trades_executed,
                        COUNT(CASE WHEN status = 'rejected' OR status = 'cancelled' THEN 1 END) as trades_failed
                    FROM trading_orders 
                    WHERE DATE(created_at) = :today
                    GROUP BY strategy
                """), {'today': today})
                
                trade_metrics = {row[0]: {'executed': row[1], 'failed': row[2]} for row in trade_result}
                
                # Save trading metrics
                for portfolio in portfolio_metrics:
                    strategy = portfolio[0]
                    total_value = portfolio[2] or 0
                    unrealized_pnl = portfolio[3] or 0
                    daily_return_pct = (unrealized_pnl / total_value * 100) if total_value > 0 else 0
                    
                    trade_data = trade_metrics.get(strategy, {'executed': 0, 'failed': 0})
                    
                    conn.execute(text("""
                        INSERT INTO trading_metrics (
                            metric_date, strategy, total_positions, total_value, 
                            daily_pnl, daily_return_pct, trades_executed, trades_failed,
                            largest_position_pct
                        ) VALUES (
                            :metric_date, :strategy, :total_positions, :total_value,
                            :daily_pnl, :daily_return_pct, :trades_executed, :trades_failed,
                            :largest_position_pct
                        )
                        ON CONFLICT (metric_date, strategy) DO UPDATE SET
                            total_positions = EXCLUDED.total_positions,
                            total_value = EXCLUDED.total_value,
                            daily_pnl = EXCLUDED.daily_pnl,
                            daily_return_pct = EXCLUDED.daily_return_pct,
                            trades_executed = EXCLUDED.trades_executed,
                            trades_failed = EXCLUDED.trades_failed,
                            largest_position_pct = EXCLUDED.largest_position_pct
                    """), {
                        'metric_date': today,
                        'strategy': strategy,
                        'total_positions': portfolio[1],
                        'total_value': total_value,
                        'daily_pnl': unrealized_pnl,
                        'daily_return_pct': daily_return_pct,
                        'trades_executed': trade_data['executed'],
                        'trades_failed': trade_data['failed'],
                        'largest_position_pct': portfolio[4] or 0
                    })
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {e}")
    
    def _check_alert_conditions(self):
        """Check conditions and generate alerts"""
        try:
            # Get latest system metrics
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT metric_name, value, threshold_warning, threshold_critical
                    FROM system_metrics
                    WHERE timestamp > NOW() - INTERVAL '5 minutes'
                    ORDER BY timestamp DESC
                    LIMIT 100
                """))
                
                recent_metrics = result.fetchall()
                
                # Check system thresholds
                for metric in recent_metrics:
                    metric_name = metric[0]
                    value = float(metric[1])
                    warning_threshold = metric[2]
                    critical_threshold = metric[3]
                    
                    # Skip if in cooldown
                    cooldown_key = f"system_{metric_name}"
                    if self._is_in_cooldown(cooldown_key):
                        continue
                    
                    # Check critical threshold
                    if critical_threshold and value >= critical_threshold:
                        alert = Alert(
                            alert_type='system_critical',
                            severity='critical',
                            title=f'Critical {metric_name.replace("_", " ").title()} Alert',
                            message=f'{metric_name.replace("_", " ").title()} is at {value:.1f}% (Critical threshold: {critical_threshold:.1f}%)',
                            data={'metric': metric_name, 'value': value, 'threshold': critical_threshold},
                            created_at=datetime.now()
                        )
                        
                        self._create_alert(alert)
                        self._set_cooldown(cooldown_key)
                        
                    # Check warning threshold
                    elif warning_threshold and value >= warning_threshold:
                        alert = Alert(
                            alert_type='system_warning',
                            severity='warning',
                            title=f'{metric_name.replace("_", " ").title()} Warning',
                            message=f'{metric_name.replace("_", " ").title()} is at {value:.1f}% (Warning threshold: {warning_threshold:.1f}%)',
                            data={'metric': metric_name, 'value': value, 'threshold': warning_threshold},
                            created_at=datetime.now()
                        )
                        
                        self._create_alert(alert)
                        self._set_cooldown(cooldown_key)
                
                # Check trading alerts
                self._check_trading_alerts(conn)
                
        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {e}")
    
    def _check_trading_alerts(self, conn):
        """Check trading-specific alert conditions"""
        try:
            # Check portfolio losses
            result = conn.execute(text("""
                SELECT strategy, daily_return_pct, largest_position_pct
                FROM trading_metrics
                WHERE metric_date = CURRENT_DATE
            """))
            
            trading_data = result.fetchall()
            
            for row in trading_data:
                strategy = row[0]
                daily_return = float(row[1]) if row[1] else 0
                largest_position = float(row[2]) if row[2] else 0
                
                # Portfolio loss alerts
                if daily_return <= self.config['portfolio_loss_critical']:
                    cooldown_key = f"portfolio_loss_{strategy}"
                    if not self._is_in_cooldown(cooldown_key):
                        alert = Alert(
                            alert_type='portfolio_loss',
                            severity='critical',
                            title='Critical Portfolio Loss',
                            message=f'{strategy} strategy has a daily loss of {daily_return:.1f}%',
                            data={'strategy': strategy, 'daily_return': daily_return},
                            created_at=datetime.now()
                        )
                        
                        self._create_alert(alert)
                        self._set_cooldown(cooldown_key)
                
                elif daily_return <= self.config['portfolio_loss_warning']:
                    cooldown_key = f"portfolio_loss_{strategy}"
                    if not self._is_in_cooldown(cooldown_key):
                        alert = Alert(
                            alert_type='portfolio_loss',
                            severity='warning',
                            title='Portfolio Loss Warning',
                            message=f'{strategy} strategy has a daily loss of {daily_return:.1f}%',
                            data={'strategy': strategy, 'daily_return': daily_return},
                            created_at=datetime.now()
                        )
                        
                        self._create_alert(alert)
                        self._set_cooldown(cooldown_key)
                
                # Position concentration alerts
                if largest_position >= self.config['position_concentration_critical']:
                    cooldown_key = f"position_concentration_{strategy}"
                    if not self._is_in_cooldown(cooldown_key):
                        alert = Alert(
                            alert_type='position_concentration',
                            severity='critical',
                            title='Critical Position Concentration',
                            message=f'{strategy} strategy has a position at {largest_position:.1f}% of portfolio',
                            data={'strategy': strategy, 'largest_position': largest_position},
                            created_at=datetime.now()
                        )
                        
                        self._create_alert(alert)
                        self._set_cooldown(cooldown_key)
                
                elif largest_position >= self.config['position_concentration_warning']:
                    cooldown_key = f"position_concentration_{strategy}"
                    if not self._is_in_cooldown(cooldown_key):
                        alert = Alert(
                            alert_type='position_concentration',
                            severity='warning',
                            title='Position Concentration Warning',
                            message=f'{strategy} strategy has a position at {largest_position:.1f}% of portfolio',
                            data={'strategy': strategy, 'largest_position': largest_position},
                            created_at=datetime.now()
                        )
                        
                        self._create_alert(alert)
                        self._set_cooldown(cooldown_key)
            
        except Exception as e:
            self.logger.error(f"Error checking trading alerts: {e}")
    
    def _health_check_loop(self):
        """Health check loop for system components"""
        while self.running:
            try:
                # Database health check
                self._check_database_health()
                
                # Trading system health check
                self._check_trading_system_health()
                
                # External API health checks
                self._check_external_apis()
                
                # Sleep for health check interval (longer than metrics)
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                time.sleep(60)
    
    def _check_database_health(self):
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            response_time = int((time.time() - start_time) * 1000)
            
            self._log_health_check('database', 'healthy', {
                'response_time_ms': response_time
            }, response_time)
            
            # Alert if database is slow
            if response_time > 1000:  # 1 second
                alert = Alert(
                    alert_type='database_performance',
                    severity='warning',
                    title='Database Performance Warning',
                    message=f'Database response time is {response_time}ms',
                    data={'response_time_ms': response_time},
                    created_at=datetime.now()
                )
                self._create_alert(alert)
            
        except Exception as e:
            self._log_health_check('database', 'unhealthy', {
                'error': str(e)
            })
            
            # Critical alert for database failure
            alert = Alert(
                alert_type='database_failure',
                severity='critical',
                title='Database Connection Failed',
                message=f'Unable to connect to database: {str(e)}',
                data={'error': str(e)},
                created_at=datetime.now()
            )
            self._create_alert(alert)
    
    def _check_trading_system_health(self):
        """Check trading system components"""
        try:
            with self.engine.connect() as conn:
                # Check recent trading activity
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM trading_orders 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """))
                
                recent_orders = result.scalar()
                
                # Check for stuck/pending orders
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM trading_orders 
                    WHERE status = 'pending' 
                    AND created_at < NOW() - INTERVAL '30 minutes'
                """))
                
                stuck_orders = result.scalar()
                
                health_data = {
                    'recent_orders_1h': recent_orders,
                    'stuck_orders': stuck_orders
                }
                
                status = 'healthy'
                if stuck_orders > 0:
                    status = 'warning'
                    
                    alert = Alert(
                        alert_type='stuck_orders',
                        severity='warning',
                        title='Stuck Orders Detected',
                        message=f'{stuck_orders} orders have been pending for over 30 minutes',
                        data=health_data,
                        created_at=datetime.now()
                    )
                    self._create_alert(alert)
                
                self._log_health_check('trading_system', status, health_data)
                
        except Exception as e:
            self.logger.error(f"Error checking trading system health: {e}")
    
    def _check_external_apis(self):
        """Check external API connectivity"""
        apis_to_check = [
            {'name': 'market_data', 'url': 'https://api.marketdata.com/health', 'timeout': 10},
            # Add other APIs as needed
        ]
        
        for api in apis_to_check:
            try:
                start_time = time.time()
                response = requests.get(api['url'], timeout=api['timeout'])
                response_time = int((time.time() - start_time) * 1000)
                
                if response.status_code == 200:
                    self._log_health_check(f"api_{api['name']}", 'healthy', {
                        'status_code': response.status_code,
                        'response_time_ms': response_time
                    }, response_time)
                else:
                    self._log_health_check(f"api_{api['name']}", 'warning', {
                        'status_code': response.status_code,
                        'response_time_ms': response_time
                    }, response_time)
                    
            except requests.exceptions.Timeout:
                self._log_health_check(f"api_{api['name']}", 'unhealthy', {
                    'error': 'timeout'
                })
            except Exception as e:
                self._log_health_check(f"api_{api['name']}", 'unhealthy', {
                    'error': str(e)
                })
    
    def _save_metrics(self, metrics: List[SystemMetric]):
        """Save metrics to database"""
        try:
            with self.engine.connect() as conn:
                for metric in metrics:
                    conn.execute(text("""
                        INSERT INTO system_metrics (
                            metric_name, value, unit, threshold_warning, 
                            threshold_critical, description, timestamp
                        ) VALUES (
                            :metric_name, :value, :unit, :threshold_warning,
                            :threshold_critical, :description, :timestamp
                        )
                    """), asdict(metric))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def _create_alert(self, alert: Alert):
        """Create and store alert"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO system_alerts (
                        alert_type, severity, title, message, data, created_at
                    ) VALUES (
                        :alert_type, :severity, :title, :message, :data, :created_at
                    )
                """), {
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'title': alert.title,
                    'message': alert.message,
                    'data': json.dumps(alert.data) if alert.data else None,
                    'created_at': alert.created_at
                })
                
                conn.commit()
            
            # Send notification if critical
            if alert.severity == 'critical':
                self._send_alert_notification(alert)
            
            self.logger.warning(f"Alert created: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification via email/SMS"""
        try:
            # Email notification
            smtp_host = os.getenv('SMTP_HOST')
            smtp_port = os.getenv('SMTP_PORT', 587)
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASSWORD')
            alert_email = os.getenv('ALERT_EMAIL')
            
            if all([smtp_host, smtp_user, smtp_password, alert_email]):
                msg = MIMEMultipart()
                msg['From'] = smtp_user
                msg['To'] = alert_email
                msg['Subject'] = f"[ACIS Alert] {alert.title}"
                
                body = f"""
                Alert Details:
                - Type: {alert.alert_type}
                - Severity: {alert.severity.upper()}
                - Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
                - Message: {alert.message}
                
                Additional Data:
                {json.dumps(alert.data, indent=2) if alert.data else 'None'}
                
                Please investigate immediately.
                
                ACIS Trading Platform Monitoring System
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP(smtp_host, smtp_port)
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
                server.quit()
                
                self.logger.info(f"Alert notification sent: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Error sending alert notification: {e}")
    
    def _log_health_check(self, component: str, status: str, details: Dict = None, response_time: int = None):
        """Log health check result"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO system_health_log (
                        component, status, details, response_time_ms
                    ) VALUES (
                        :component, :status, :details, :response_time_ms
                    )
                """), {
                    'component': component,
                    'status': status,
                    'details': json.dumps(details) if details else None,
                    'response_time_ms': response_time
                })
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging health check: {e}")
    
    def _is_in_cooldown(self, key: str) -> bool:
        """Check if alert is in cooldown period"""
        if key in self.alert_cooldowns:
            cooldown_end = self.alert_cooldowns[key]
            if datetime.now() < cooldown_end:
                return True
            else:
                del self.alert_cooldowns[key]
        
        return False
    
    def _set_cooldown(self, key: str):
        """Set cooldown period for alert"""
        cooldown_minutes = self.config['alert_cooldown_minutes']
        self.alert_cooldowns[key] = datetime.now() + timedelta(minutes=cooldown_minutes)
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['metrics_retention_days'])
            
            with self.engine.connect() as conn:
                # Clean system metrics
                conn.execute(text("""
                    DELETE FROM system_metrics 
                    WHERE timestamp < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                # Clean health logs
                conn.execute(text("""
                    DELETE FROM system_health_log 
                    WHERE timestamp < :cutoff_date
                """), {'cutoff_date': cutoff_date})
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error cleaning old metrics: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status summary"""
        try:
            with self.engine.connect() as conn:
                # Get latest metrics
                result = conn.execute(text("""
                    SELECT metric_name, value, unit
                    FROM system_metrics
                    WHERE timestamp > NOW() - INTERVAL '5 minutes'
                    ORDER BY timestamp DESC
                """))
                
                metrics = {row[0]: {'value': float(row[1]), 'unit': row[2]} for row in result}
                
                # Get active alerts
                result = conn.execute(text("""
                    SELECT COUNT(*) as alert_count, severity
                    FROM system_alerts
                    WHERE acknowledged = false
                    GROUP BY severity
                """))
                
                alerts = {row[1]: row[0] for row in result}
                
                # Get system health
                result = conn.execute(text("""
                    SELECT component, status
                    FROM system_health_log
                    WHERE timestamp > NOW() - INTERVAL '10 minutes'
                    GROUP BY component, status
                    ORDER BY MAX(timestamp) DESC
                """))
                
                health_status = {row[0]: row[1] for row in result}
                
                return {
                    'metrics': metrics,
                    'alerts': alerts,
                    'health': health_status,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}

def main():
    """Run system monitoring"""
    print("ACIS Trading Platform - System Monitoring")
    print("Starting comprehensive system monitoring...")
    
    monitor = SystemMonitor()
    monitor.start_monitoring()
    
    try:
        # Keep monitoring running
        while True:
            status = monitor.get_system_status()
            print(f"\nSystem Status Update: {datetime.now()}")
            print(f"Active Alerts: {sum(status.get('alerts', {}).values())}")
            print(f"System Health Components: {len(status.get('health', {}))}")
            
            # Sleep for status update
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        print("\nShutting down monitoring...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()