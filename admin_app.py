#!/usr/bin/env python3
"""
ACIS Trading Platform - Comprehensive Admin Application
Full-featured web interface for managing strategies, portfolios, trades, clients, and system operations
"""

import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, SelectField, IntegerField, FloatField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, Email, NumberRange
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, text, func
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json

# Load environment
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'acis-admin-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('POSTGRES_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
csrf = CSRFProtect(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access the admin panel.'

# Database connection
engine = create_engine(os.getenv('POSTGRES_URL'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ACIS_Admin')

# User roles
ROLES = {
    'admin': 'System Administrator',
    'portfolio_manager': 'Portfolio Manager', 
    'trader': 'Trader',
    'analyst': 'Research Analyst',
    'client_service': 'Client Service',
    'risk_manager': 'Risk Manager',
    'readonly': 'Read Only'
}

# User model
class User(UserMixin):
    def __init__(self, user_id, username, email, role, full_name):
        self.id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.full_name = full_name
    
    def has_permission(self, permission):
        """Check if user has specific permission based on role"""
        permissions = {
            'admin': ['all'],
            'portfolio_manager': ['view_portfolios', 'manage_portfolios', 'view_trades', 'manage_strategies'],
            'trader': ['view_portfolios', 'view_trades', 'execute_trades', 'view_positions'],
            'analyst': ['view_portfolios', 'view_trades', 'view_strategies', 'manage_research'],
            'client_service': ['view_portfolios', 'view_clients', 'manage_clients', 'generate_reports'],
            'risk_manager': ['view_portfolios', 'view_trades', 'manage_risk', 'view_all'],
            'readonly': ['view_portfolios', 'view_trades', 'view_clients']
        }
        
        user_permissions = permissions.get(self.role, [])
        return permission in user_permissions or 'all' in user_permissions

@login_manager.user_loader
def load_user(user_id):
    """Load user from database"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, username, email, role, full_name 
                FROM admin_users 
                WHERE id = :user_id AND active = true
            """), {'user_id': user_id})
            
            user_data = result.fetchone()
            if user_data:
                return User(user_data[0], user_data[1], user_data[2], user_data[3], user_data[4])
    except Exception as e:
        logger.error(f"Error loading user: {e}")
    
    return None

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])

class UserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    full_name = StringField('Full Name', validators=[DataRequired()])
    role = SelectField('Role', choices=[(k, v) for k, v in ROLES.items()])
    password = PasswordField('Password')
    active = BooleanField('Active')

class StrategyForm(FlaskForm):
    name = StringField('Strategy Name', validators=[DataRequired()])
    description = TextAreaField('Description')
    market_cap = SelectField('Market Cap', choices=[('small', 'Small Cap'), ('mid', 'Mid Cap'), ('large', 'Large Cap')])
    style = SelectField('Style', choices=[('value', 'Value'), ('growth', 'Growth'), ('momentum', 'Momentum'), ('dividend', 'Dividend')])
    max_positions = IntegerField('Max Positions', validators=[NumberRange(min=10, max=200)])
    target_turnover = FloatField('Target Turnover %', validators=[NumberRange(min=0, max=500)])
    active = BooleanField('Active')

class ClientForm(FlaskForm):
    name = StringField('Client Name', validators=[DataRequired()])
    email = StringField('Email', validators=[Email()])
    type = SelectField('Client Type', choices=[('individual', 'Individual'), ('institutional', 'Institutional')])
    investment_amount = FloatField('Investment Amount', validators=[NumberRange(min=0)])
    risk_tolerance = SelectField('Risk Tolerance', choices=[('conservative', 'Conservative'), ('moderate', 'Moderate'), ('aggressive', 'Aggressive')])
    strategies = StringField('Assigned Strategies (comma-separated)')
    active = BooleanField('Active')

# Initialize database tables
def init_admin_tables():
    """Initialize admin database tables"""
    with engine.connect() as conn:
        # Admin users table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                role VARCHAR(20) NOT NULL,
                active BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """))
        
        # System settings table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS system_settings (
                id SERIAL PRIMARY KEY,
                setting_key VARCHAR(50) UNIQUE NOT NULL,
                setting_value TEXT,
                description TEXT,
                updated_by INTEGER REFERENCES admin_users(id),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Strategy configurations table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS strategy_configurations (
                id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(100) NOT NULL,
                config_key VARCHAR(100) NOT NULL,
                config_value TEXT,
                description TEXT,
                updated_by INTEGER REFERENCES admin_users(id),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_name, config_key)
            )
        """))
        
        # Client management table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS clients (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE,
                client_type VARCHAR(20) NOT NULL,
                investment_amount DECIMAL(15,2),
                risk_tolerance VARCHAR(20),
                assigned_strategies TEXT[],
                active BOOLEAN DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER REFERENCES admin_users(id)
            )
        """))
        
        # Audit log table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES admin_users(id),
                action VARCHAR(100) NOT NULL,
                table_name VARCHAR(50),
                record_id INTEGER,
                old_values JSONB,
                new_values JSONB,
                ip_address INET,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # System alerts table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS system_alerts (
                id SERIAL PRIMARY KEY,
                alert_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                title VARCHAR(200) NOT NULL,
                message TEXT,
                data JSONB,
                acknowledged BOOLEAN DEFAULT false,
                acknowledged_by INTEGER REFERENCES admin_users(id),
                acknowledged_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.commit()
    
    # Create default admin user if none exists
    create_default_admin()

def create_default_admin():
    """Create default admin user if none exists"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM admin_users WHERE role = 'admin'"))
            admin_count = result.scalar()
            
            if admin_count == 0:
                password_hash = generate_password_hash('admin123')
                conn.execute(text("""
                    INSERT INTO admin_users (username, email, password_hash, full_name, role)
                    VALUES ('admin', 'admin@acis.com', :password_hash, 'System Administrator', 'admin')
                """), {'password_hash': password_hash})
                conn.commit()
                logger.info("Created default admin user (username: admin, password: admin123)")
    
    except Exception as e:
        logger.error(f"Error creating default admin: {e}")

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, username, email, password_hash, role, full_name
                    FROM admin_users 
                    WHERE username = :username AND active = true
                """), {'username': username})
                
                user_data = result.fetchone()
                
                if user_data and check_password_hash(user_data[3], password):
                    user = User(user_data[0], user_data[1], user_data[2], user_data[4], user_data[5])
                    login_user(user)
                    
                    # Update last login
                    conn.execute(text("""
                        UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE id = :user_id
                    """), {'user_id': user.id})
                    conn.commit()
                    
                    # Log successful login
                    log_audit_event('user_login', 'admin_users', user.id)
                    
                    flash(f'Welcome back, {user.full_name}!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username or password', 'danger')
        
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('Login system error', 'danger')
    
    return render_template('admin/login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    log_audit_event('user_logout', 'admin_users', current_user.id)
    logout_user()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('login'))

# Main dashboard
@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    """Main admin dashboard"""
    try:
        # Get system overview data
        dashboard_data = get_dashboard_data()
        return render_template('admin/dashboard.html', data=dashboard_data)
    
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash('Error loading dashboard data', 'danger')
        return render_template('admin/dashboard.html', data={})

def get_dashboard_data():
    """Get dashboard overview data"""
    data = {}
    
    try:
        with engine.connect() as conn:
            # Portfolio overview
            result = conn.execute(text("""
                SELECT 
                    COUNT(DISTINCT symbol) as total_positions,
                    SUM(market_value) as total_value,
                    AVG(unrealized_pnl/market_value*100) as avg_return
                FROM trading_positions 
                WHERE quantity > 0
            """))
            portfolio_data = result.fetchone()
            
            data['portfolio'] = {
                'total_positions': portfolio_data[0] or 0,
                'total_value': portfolio_data[1] or 0,
                'avg_return': portfolio_data[2] or 0
            }
            
            # Recent trades
            result = conn.execute(text("""
                SELECT COUNT(*) as trade_count,
                       SUM(CASE WHEN side = 'buy' THEN quantity * avg_fill_price ELSE 0 END) as total_bought,
                       SUM(CASE WHEN side = 'sell' THEN quantity * avg_fill_price ELSE 0 END) as total_sold
                FROM trading_orders 
                WHERE status = 'filled' AND filled_at >= CURRENT_DATE - INTERVAL '30 days'
            """))
            trade_data = result.fetchone()
            
            data['trades'] = {
                'count': trade_data[0] or 0,
                'total_bought': trade_data[1] or 0,
                'total_sold': trade_data[2] or 0
            }
            
            # System alerts
            result = conn.execute(text("""
                SELECT COUNT(*) as alert_count
                FROM system_alerts 
                WHERE acknowledged = false
            """))
            alert_count = result.scalar()
            
            data['alerts'] = {'count': alert_count or 0}
            
            # Active clients
            result = conn.execute(text("""
                SELECT COUNT(*) as client_count
                FROM clients 
                WHERE active = true
            """))
            client_count = result.scalar()
            
            data['clients'] = {'count': client_count or 0}
    
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        data = {'portfolio': {}, 'trades': {}, 'alerts': {}, 'clients': {}}
    
    return data

# User management routes
@app.route('/users')
@login_required
def users():
    """User management page"""
    if not current_user.has_permission('manage_users') and current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, username, email, full_name, role, active, created_at, last_login
                FROM admin_users 
                ORDER BY created_at DESC
            """))
            users_data = result.fetchall()
        
        return render_template('admin/users.html', users=users_data, roles=ROLES)
    
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        flash('Error loading users', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/users/add', methods=['GET', 'POST'])
@login_required
def add_user():
    """Add new user"""
    if not current_user.has_permission('manage_users') and current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('users'))
    
    form = UserForm()
    if form.validate_on_submit():
        try:
            password_hash = generate_password_hash(form.password.data)
            
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO admin_users (username, email, password_hash, full_name, role, active)
                    VALUES (:username, :email, :password_hash, :full_name, :role, :active)
                """), {
                    'username': form.username.data,
                    'email': form.email.data,
                    'password_hash': password_hash,
                    'full_name': form.full_name.data,
                    'role': form.role.data,
                    'active': form.active.data
                })
                conn.commit()
            
            log_audit_event('user_created', 'admin_users', None, {'username': form.username.data})
            flash('User created successfully', 'success')
            return redirect(url_for('users'))
        
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            flash('Error creating user', 'danger')
    
    return render_template('admin/user_form.html', form=form, title='Add User')

def log_audit_event(action, table_name, record_id, new_values=None, old_values=None):
    """Log audit event"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO audit_log (user_id, action, table_name, record_id, old_values, new_values, ip_address)
                VALUES (:user_id, :action, :table_name, :record_id, :old_values, :new_values, :ip_address)
            """), {
                'user_id': current_user.id if current_user.is_authenticated else None,
                'action': action,
                'table_name': table_name,
                'record_id': record_id,
                'old_values': json.dumps(old_values) if old_values else None,
                'new_values': json.dumps(new_values) if new_values else None,
                'ip_address': request.remote_addr
            })
            conn.commit()
    
    except Exception as e:
        logger.error(f"Error logging audit event: {e}")

# Strategy management routes
@app.route('/strategies')
@login_required
def strategies():
    """Strategy management page"""
    try:
        # Get strategy performance data
        strategy_data = get_strategy_overview()
        return render_template('admin/strategies.html', strategies=strategy_data)
    
    except Exception as e:
        logger.error(f"Error loading strategies: {e}")
        flash('Error loading strategies', 'danger')
        return redirect(url_for('dashboard'))

def get_strategy_overview():
    """Get strategy overview data"""
    strategies = []
    
    strategy_names = [
        'small_cap_value', 'small_cap_growth', 'small_cap_momentum', 'small_cap_dividend',
        'mid_cap_value', 'mid_cap_growth', 'mid_cap_momentum', 'mid_cap_dividend',
        'large_cap_value', 'large_cap_growth', 'large_cap_momentum', 'large_cap_dividend'
    ]
    
    try:
        with engine.connect() as conn:
            for strategy in strategy_names:
                # Get portfolio metrics
                result = conn.execute(text(f"""
                    SELECT 
                        COUNT(*) as positions,
                        SUM(market_value) as total_value,
                        AVG(unrealized_pnl/market_value*100) as avg_return
                    FROM trading_positions 
                    WHERE strategy = :strategy AND quantity > 0
                """), {'strategy': strategy})
                
                data = result.fetchone()
                
                strategies.append({
                    'name': strategy,
                    'display_name': strategy.replace('_', ' ').title(),
                    'positions': data[0] or 0,
                    'value': data[1] or 0,
                    'return': data[2] or 0,
                    'active': True
                })
    
    except Exception as e:
        logger.error(f"Error getting strategy overview: {e}")
    
    return strategies

# Portfolio management routes
@app.route('/portfolios')
@login_required  
def portfolios():
    """Portfolio management page"""
    try:
        portfolio_data = get_portfolio_overview()
        return render_template('admin/portfolios.html', portfolios=portfolio_data)
    
    except Exception as e:
        logger.error(f"Error loading portfolios: {e}")
        flash('Error loading portfolios', 'danger')
        return redirect(url_for('dashboard'))

def get_portfolio_overview():
    """Get portfolio overview data"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    strategy,
                    COUNT(*) as positions,
                    SUM(quantity) as total_shares,
                    SUM(market_value) as market_value,
                    SUM(unrealized_pnl) as unrealized_pnl,
                    SUM(realized_pnl) as realized_pnl
                FROM trading_positions 
                WHERE quantity > 0
                GROUP BY strategy
                ORDER BY market_value DESC
            """))
            
            portfolios = []
            for row in result:
                portfolios.append({
                    'strategy': row[0],
                    'display_name': row[0].replace('_', ' ').title() if row[0] else 'Unknown',
                    'positions': row[1],
                    'total_shares': row[2],
                    'market_value': row[3],
                    'unrealized_pnl': row[4],
                    'realized_pnl': row[5],
                    'total_pnl': (row[4] or 0) + (row[5] or 0)
                })
            
            return portfolios
    
    except Exception as e:
        logger.error(f"Error getting portfolio overview: {e}")
        return []

if __name__ == '__main__':
    # Initialize database tables
    init_admin_tables()
    
    # Run the application
    port = int(os.getenv('ADMIN_PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting ACIS Trading Platform Admin Interface on port {port}")
    print(f"Default login: admin / admin123")
    
    app.run(host='0.0.0.0', port=port, debug=debug)