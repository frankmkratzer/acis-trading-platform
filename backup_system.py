#!/usr/bin/env python3
"""
ACIS Trading Platform - Automated Backup System
Handles database backups, file backups, and retention management
"""

import os
import subprocess
import datetime
import logging
import time
import schedule
from pathlib import Path
import json
import gzip
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ACISBackupSystem:
    def __init__(self):
        self.postgres_url = os.environ.get('POSTGRES_URL')
        self.backup_dir = Path('/app/backups')
        self.retention_days = int(os.environ.get('BACKUP_RETENTION_DAYS', '30'))
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ACIS Backup System initialized")
    
    def create_database_backup(self):
        """Create PostgreSQL database backup"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"acis_trading_{timestamp}.sql"
            
            # Extract connection details from URL
            # postgresql://user:password@host:port/database
            if self.postgres_url.startswith('postgresql://'):
                url_parts = self.postgres_url.replace('postgresql://', '').split('@')
                user_pass = url_parts[0].split(':')
                host_db = url_parts[1].split('/')
                host_port = host_db[0].split(':')
                
                user = user_pass[0]
                password = user_pass[1]
                host = host_port[0]
                port = host_port[1] if len(host_port) > 1 else '5432'
                database = host_db[1]
                
                # Set password environment variable
                env = os.environ.copy()
                env['PGPASSWORD'] = password
                
                # Run pg_dump
                cmd = [
                    'pg_dump',
                    '-h', host,
                    '-p', port,
                    '-U', user,
                    '-d', database,
                    '--verbose',
                    '--clean',
                    '--if-exists',
                    '--create',
                    '--format=plain'
                ]
                
                with open(backup_file, 'w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                          env=env, text=True)
                
                if result.returncode == 0:
                    # Compress the backup
                    compressed_file = f"{backup_file}.gz"
                    with open(backup_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove uncompressed file
                    os.remove(backup_file)
                    
                    size_mb = os.path.getsize(compressed_file) / 1024 / 1024
                    logger.info(f"Database backup created: {compressed_file} ({size_mb:.1f} MB)")
                    
                    return compressed_file
                else:
                    logger.error(f"Database backup failed: {result.stderr}")
                    return None
            else:
                logger.error("Invalid PostgreSQL URL format")
                return None
                
        except Exception as e:
            logger.error(f"Database backup error: {str(e)}")
            return None
    
    def create_files_backup(self):
        """Create backup of important files"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"acis_files_{timestamp}.tar.gz"
            
            # Files and directories to backup
            files_to_backup = [
                '/app/reports',
                '/app/logs',
                '/app/config.py',
                '/app/database_config.py'
            ]
            
            # Create tar.gz archive
            cmd = ['tar', '-czf', str(backup_file)]
            for file_path in files_to_backup:
                if os.path.exists(file_path):
                    cmd.extend(['-C', '/', file_path.lstrip('/')])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                size_mb = os.path.getsize(backup_file) / 1024 / 1024
                logger.info(f"Files backup created: {backup_file} ({size_mb:.1f} MB)")
                return backup_file
            else:
                logger.error(f"Files backup failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Files backup error: {str(e)}")
            return None
    
    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.retention_days)
            deleted_count = 0
            
            for backup_file in self.backup_dir.glob('*'):
                if backup_file.is_file():
                    file_date = datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        backup_file.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old backup: {backup_file.name}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old backup files")
            
        except Exception as e:
            logger.error(f"Backup cleanup error: {str(e)}")
    
    def create_backup_manifest(self, db_backup, files_backup):
        """Create manifest file with backup information"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            manifest_file = self.backup_dir / f"backup_manifest_{timestamp}.json"
            
            manifest = {
                'timestamp': timestamp,
                'database_backup': str(db_backup) if db_backup else None,
                'files_backup': str(files_backup) if files_backup else None,
                'backup_date': datetime.datetime.now().isoformat(),
                'retention_days': self.retention_days
            }
            
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Backup manifest created: {manifest_file}")
            
        except Exception as e:
            logger.error(f"Manifest creation error: {str(e)}")
    
    def run_backup(self):
        """Run complete backup process"""
        logger.info("Starting ACIS backup process...")
        
        # Create database backup
        db_backup = self.create_database_backup()
        
        # Create files backup
        files_backup = self.create_files_backup()
        
        # Create manifest
        self.create_backup_manifest(db_backup, files_backup)
        
        # Cleanup old backups
        self.cleanup_old_backups()
        
        logger.info("ACIS backup process completed")
    
    def run_scheduler(self):
        """Run backup scheduler"""
        backup_schedule = os.environ.get('BACKUP_SCHEDULE', '0 2 * * *')  # Default: 2 AM daily
        
        # Parse cron expression (simplified - daily at specified hour)
        if backup_schedule == '0 2 * * *':  # Daily at 2 AM
            schedule.every().day.at("02:00").do(self.run_backup)
        elif backup_schedule == '0 */6 * * *':  # Every 6 hours
            schedule.every(6).hours.do(self.run_backup)
        elif backup_schedule == '0 * * * *':  # Every hour
            schedule.every().hour.do(self.run_backup)
        else:
            # Default to daily at 2 AM
            schedule.every().day.at("02:00").do(self.run_backup)
        
        logger.info(f"Backup scheduler started with schedule: {backup_schedule}")
        
        # Run initial backup
        self.run_backup()
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main backup service entry point"""
    logger.info("Starting ACIS Backup System...")
    
    backup_system = ACISBackupSystem()
    
    try:
        backup_system.run_scheduler()
    except KeyboardInterrupt:
        logger.info("Backup system shutting down...")
    except Exception as e:
        logger.error(f"Backup system error: {str(e)}")
        raise

if __name__ == "__main__":
    main()