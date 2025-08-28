#!/usr/bin/env python3
"""
ACIS Monitoring Dashboard
Comprehensive monitoring and alerting for ACIS trading platform reliability
Shows health metrics, error rates, data quality, and system performance
"""

import os
import sys
import time
import json
import smtplib
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from logging_config import setup_logger

load_dotenv()
logger = setup_logger("monitoring_dashboard")

POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    logger.error("POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

class ACISMonitoring:
    """Comprehensive monitoring for ACIS platform"""
    
    def __init__(self):
        self.today = date.today()
        self.yesterday = self.today - timedelta(days=1)
    
    def get_script_health_summary(self, days: int = 7) -> Dict:
        """Get script health summary for last N days"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        script_name,
                        COUNT(*) as total_runs,
                        COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as successful_runs,
                        COUNT(CASE WHEN status = 'FAILED' THEN 1 END) as failed_runs,
                        COUNT(CASE WHEN status = 'PARTIAL_SUCCESS' THEN 1 END) as partial_runs,
                        AVG(execution_time_seconds) as avg_execution_time,
                        AVG(memory_usage_mb) as avg_memory_usage,
                        MAX(created_at) as last_run_time
                    FROM script_health_monitor 
                    WHERE created_at >= :cutoff_date
                    GROUP BY script_name
                    ORDER BY script_name
                """), {'cutoff_date': self.today - timedelta(days=days)})
                
                health_data = {}
                for row in result:
                    health_data[row[0]] = {
                        'total_runs': row[1],
                        'successful_runs': row[2],
                        'failed_runs': row[3],
                        'partial_runs': row[4],
                        'success_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0,
                        'avg_execution_time': float(row[5]) if row[5] else 0,
                        'avg_memory_usage': float(row[6]) if row[6] else 0,
                        'last_run_time': row[7]
                    }
                
                return health_data
                
        except Exception as e:
            logger.error(f"Failed to get script health summary: {e}")
            return {}
    
    def get_error_summary(self, days: int = 7) -> Dict:
        """Get error summary for last N days"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        script_name,
                        error_type,
                        COUNT(*) as error_count,
                        COUNT(CASE WHEN resolved = TRUE THEN 1 END) as resolved_count,
                        MAX(occurred_at) as latest_error
                    FROM error_tracking 
                    WHERE occurred_at >= :cutoff_date
                    GROUP BY script_name, error_type
                    ORDER BY error_count DESC
                """), {'cutoff_date': self.today - timedelta(days=days)})
                
                errors = []
                for row in result:
                    errors.append({
                        'script_name': row[0],
                        'error_type': row[1],
                        'error_count': row[2],
                        'resolved_count': row[3],
                        'unresolved_count': row[2] - row[3],
                        'latest_error': row[4]
                    })
                
                return errors
                
        except Exception as e:
            logger.error(f"Failed to get error summary: {e}")
            return []
    
    def get_data_quality_summary(self, days: int = 7) -> Dict:
        """Get data quality summary for last N days"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        table_name,
                        check_type,
                        COUNT(*) as total_checks,
                        COUNT(CASE WHEN passed = TRUE THEN 1 END) as passed_checks,
                        MAX(created_at) as latest_check
                    FROM data_quality_checks 
                    WHERE created_at >= :cutoff_date
                    GROUP BY table_name, check_type
                    ORDER BY table_name, check_type
                """), {'cutoff_date': self.today - timedelta(days=days)})
                
                quality_data = {}
                for row in result:
                    table = row[0]
                    if table not in quality_data:
                        quality_data[table] = {}
                    
                    quality_data[table][row[1]] = {
                        'total_checks': row[2],
                        'passed_checks': row[3],
                        'failed_checks': row[2] - row[3],
                        'pass_rate': (row[3] / row[2] * 100) if row[2] > 0 else 0,
                        'latest_check': row[4]
                    }
                
                return quality_data
                
        except Exception as e:
            logger.error(f"Failed to get data quality summary: {e}")
            return {}
    
    def get_circuit_breaker_status(self) -> Dict:
        """Check circuit breaker status from reliability manager"""
        try:
            from reliability_manager import api_circuit_breakers
            
            status = {}
            for api_name, breaker in api_circuit_breakers.items():
                status[api_name] = {
                    'state': breaker.state,
                    'failure_count': breaker.failure_count,
                    'last_failure_time': breaker.last_failure_time,
                    'is_open': breaker.state == 'OPEN'
                }
            
            return status
            
        except Exception as e:
            logger.warning(f"Could not get circuit breaker status: {e}")
            return {}
    
    def get_incremental_fetch_status(self) -> Dict:
        """Get incremental fetch manager status"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        data_type,
                        fetch_status,
                        COUNT(*) as symbol_count,
                        MAX(last_fetch_date) as most_recent_fetch,
                        MIN(last_fetch_date) as oldest_fetch
                    FROM fetch_status_tracking 
                    GROUP BY data_type, fetch_status
                    ORDER BY data_type, fetch_status
                """))
                
                fetch_data = {}
                for row in result:
                    data_type = row[0]
                    if data_type not in fetch_data:
                        fetch_data[data_type] = {}
                    
                    fetch_data[data_type][row[1]] = {
                        'symbol_count': row[2],
                        'most_recent': row[3],
                        'oldest': row[4]
                    }
                
                return fetch_data
                
        except Exception as e:
            logger.error(f"Failed to get incremental fetch status: {e}")
            return {}
    
    def check_critical_alerts(self) -> List[Dict]:
        """Check for critical issues that need immediate attention"""
        alerts = []
        
        # Check for scripts that haven't run recently
        health = self.get_script_health_summary(days=2)
        for script_name, data in health.items():
            if data['last_run_time']:
                hours_since_run = (datetime.now() - data['last_run_time']).total_seconds() / 3600
                if hours_since_run > 48:  # No run in 48 hours
                    alerts.append({
                        'severity': 'HIGH',
                        'type': 'STALE_SCRIPT',
                        'message': f"{script_name} hasn't run in {hours_since_run:.1f} hours",
                        'script': script_name
                    })
        
        # Check for high error rates
        for script_name, data in health.items():
            if data['success_rate'] < 50 and data['total_runs'] > 3:  # Less than 50% success rate
                alerts.append({
                    'severity': 'HIGH',
                    'type': 'HIGH_ERROR_RATE',
                    'message': f"{script_name} has {data['success_rate']:.1f}% success rate",
                    'script': script_name
                })
        
        # Check circuit breakers
        circuit_status = self.get_circuit_breaker_status()
        for api_name, status in circuit_status.items():
            if status['is_open']:
                alerts.append({
                    'severity': 'HIGH',
                    'type': 'CIRCUIT_BREAKER_OPEN',
                    'message': f"{api_name} circuit breaker is OPEN",
                    'api': api_name
                })
        
        # Check for unresolved errors
        errors = self.get_error_summary(days=1)
        for error in errors:
            if error['unresolved_count'] > 10:  # More than 10 unresolved errors
                alerts.append({
                    'severity': 'MEDIUM',
                    'type': 'UNRESOLVED_ERRORS',
                    'message': f"{error['script_name']} has {error['unresolved_count']} unresolved {error['error_type']} errors",
                    'script': error['script_name']
                })
        
        return alerts
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        report = []
        report.append("=" * 80)
        report.append("🏥 ACIS PLATFORM HEALTH REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Critical alerts
        alerts = self.check_critical_alerts()
        if alerts:
            report.append("🚨 CRITICAL ALERTS:")
            for alert in alerts:
                severity_emoji = "🔴" if alert['severity'] == 'HIGH' else "🟡"
                report.append(f"  {severity_emoji} {alert['type']}: {alert['message']}")
            report.append("")
        else:
            report.append("✅ No critical alerts")
            report.append("")
        
        # Script health summary
        health = self.get_script_health_summary(days=7)
        if health:
            report.append("📊 SCRIPT HEALTH (Last 7 days):")
            report.append("-" * 80)
            for script_name, data in health.items():
                status_emoji = "✅" if data['success_rate'] > 90 else "⚠️" if data['success_rate'] > 50 else "❌"
                report.append(f"  {status_emoji} {script_name}:")
                report.append(f"    Success Rate: {data['success_rate']:.1f}% ({data['successful_runs']}/{data['total_runs']})")
                report.append(f"    Avg Execution: {data['avg_execution_time']:.1f}s")
                report.append(f"    Avg Memory: {data['avg_memory_usage']:.1f}MB")
                report.append(f"    Last Run: {data['last_run_time']}")
                report.append("")
        
        # Error summary
        errors = self.get_error_summary(days=7)
        if errors:
            report.append("⚠️  ERROR SUMMARY (Last 7 days):")
            report.append("-" * 80)
            for error in errors[:10]:  # Show top 10 errors
                report.append(f"  {error['script_name']} - {error['error_type']}: {error['error_count']} errors ({error['unresolved_count']} unresolved)")
            report.append("")
        
        # Data quality summary
        quality = self.get_data_quality_summary(days=7)
        if quality:
            report.append("🔍 DATA QUALITY (Last 7 days):")
            report.append("-" * 80)
            for table_name, checks in quality.items():
                report.append(f"  {table_name}:")
                for check_type, data in checks.items():
                    quality_emoji = "✅" if data['pass_rate'] > 95 else "⚠️" if data['pass_rate'] > 80 else "❌"
                    report.append(f"    {quality_emoji} {check_type}: {data['pass_rate']:.1f}% pass rate ({data['passed_checks']}/{data['total_checks']})")
                report.append("")
        
        # Circuit breaker status
        circuit_status = self.get_circuit_breaker_status()
        if circuit_status:
            report.append("🔌 CIRCUIT BREAKER STATUS:")
            report.append("-" * 80)
            for api_name, status in circuit_status.items():
                state_emoji = "✅" if status['state'] == 'CLOSED' else "⚠️" if status['state'] == 'HALF_OPEN' else "❌"
                report.append(f"  {state_emoji} {api_name}: {status['state']} (failures: {status['failure_count']})")
            report.append("")
        
        # Incremental fetch status
        fetch_status = self.get_incremental_fetch_status()
        if fetch_status:
            report.append("📥 INCREMENTAL FETCH STATUS:")
            report.append("-" * 80)
            for data_type, statuses in fetch_status.items():
                report.append(f"  {data_type}:")
                for status, data in statuses.items():
                    report.append(f"    {status}: {data['symbol_count']} symbols (latest: {data['most_recent']})")
                report.append("")
        
        report.append("=" * 80)
        report.append("End of Health Report")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def send_alert_email(self, subject: str, body: str):
        """Send alert email if configured"""
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        alert_emails = os.getenv("ALERT_EMAILS", "").split(",")
        
        if not all([smtp_server, smtp_user, smtp_password]) or not alert_emails[0]:
            logger.warning("Email alerting not configured - skipping email notification")
            return False
        
        try:
            msg = MimeMultipart()
            msg['From'] = smtp_user
            msg['To'] = ", ".join(alert_emails)
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alert email sent to {len(alert_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ACIS Platform Monitoring Dashboard")
    parser.add_argument("--report", action="store_true", help="Generate and display health report")
    parser.add_argument("--alerts", action="store_true", help="Check for critical alerts")
    parser.add_argument("--email", action="store_true", help="Send email alerts if critical issues found")
    parser.add_argument("--days", type=int, default=7, help="Number of days to include in reports")
    
    args = parser.parse_args()
    
    monitor = ACISMonitoring()
    
    if args.alerts or args.email:
        alerts = monitor.check_critical_alerts()
        
        if alerts:
            print(f"🚨 Found {len(alerts)} critical alerts:")
            for alert in alerts:
                severity_emoji = "🔴" if alert['severity'] == 'HIGH' else "🟡"
                print(f"  {severity_emoji} {alert['type']}: {alert['message']}")
            
            if args.email:
                subject = f"ACIS Platform Alert - {len(alerts)} Critical Issues"
                body = "Critical issues detected in ACIS platform:\n\n"
                for alert in alerts:
                    body += f"• {alert['type']}: {alert['message']}\n"
                body += f"\nGenerated: {datetime.now()}\n"
                
                monitor.send_alert_email(subject, body)
        else:
            print("✅ No critical alerts found")
    
    if args.report:
        report = monitor.generate_health_report()
        print(report)
        
        # Save report to file
        report_file = f"logs/health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("logs", exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_file}")

if __name__ == "__main__":
    main()