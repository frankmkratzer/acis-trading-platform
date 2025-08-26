#!/usr/bin/env python3
"""
ACIS Client Report Generator
Professional PDF portfolio reports for clients
Comprehensive performance analysis and presentation
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

class ClientReportGenerator:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(os.getenv('POSTGRES_URL'))
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Report configuration
        self.report_config = {
            'company_name': 'ACIS Trading Platform',
            'logo_path': None,  # Path to company logo
            'colors': {
                'primary': '#2c3e50',
                'secondary': '#3498db',
                'success': '#27ae60',
                'danger': '#e74c3c',
                'warning': '#f39c12'
            },
            'benchmark': 'S&P 500'
        }
        
    def generate_client_report(self, client_name: str, strategy: str = "all", 
                             report_period: str = "1Y", output_path: str = None) -> str:
        """Generate comprehensive client portfolio report"""
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"client_report_{client_name}_{strategy}_{timestamp}.pdf"
        
        print(f"Generating client report for {client_name}...")
        print(f"Strategy: {strategy}, Period: {report_period}")
        
        with PdfPages(output_path) as pdf:
            # Page 1: Cover Page
            self._create_cover_page(pdf, client_name, strategy, report_period)
            
            # Page 2: Executive Summary
            self._create_executive_summary(pdf, strategy, report_period)
            
            # Page 3: Performance Overview
            self._create_performance_overview(pdf, strategy, report_period)
            
            # Page 4: Portfolio Allocation
            self._create_allocation_analysis(pdf, strategy)
            
            # Page 5: Holdings Analysis
            self._create_holdings_analysis(pdf, strategy)
            
            # Page 6: Risk Analysis
            self._create_risk_analysis(pdf, strategy, report_period)
            
            # Page 7: Strategy Performance
            self._create_strategy_performance(pdf, strategy)
            
            # Page 8: Market Commentary
            self._create_market_commentary(pdf, report_period)
            
            # Page 9: Appendix
            self._create_appendix(pdf, strategy)
        
        print(f"Client report generated: {output_path}")
        return output_path
    
    def _create_cover_page(self, pdf, client_name: str, strategy: str, period: str):
        """Create professional cover page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Company header
        ax.text(5, 12, self.report_config['company_name'], 
                ha='center', va='center', fontsize=24, fontweight='bold',
                color=self.report_config['colors']['primary'])
        
        ax.text(5, 11.3, 'Portfolio Performance Report', 
                ha='center', va='center', fontsize=16,
                color=self.report_config['colors']['secondary'])
        
        # Horizontal line
        ax.axhline(y=10.8, xmin=0.1, xmax=0.9, color=self.report_config['colors']['primary'], linewidth=2)
        
        # Client information
        ax.text(5, 9.5, f'Prepared for: {client_name}', 
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        ax.text(5, 8.8, f'Strategy: {strategy.replace("_", " ").title()}', 
                ha='center', va='center', fontsize=14)
        
        ax.text(5, 8.3, f'Reporting Period: {period}', 
                ha='center', va='center', fontsize=14)
        
        # Report date
        ax.text(5, 7.5, f'Report Date: {datetime.now().strftime("%B %d, %Y")}', 
                ha='center', va='center', fontsize=12)
        
        # Disclaimer box
        disclaimer_text = (
            "IMPORTANT DISCLAIMER\n\n"
            "This report is for informational purposes only and does not constitute investment advice.\n"
            "Past performance does not guarantee future results. All investments carry risk of loss.\n"
            "Please consult with your financial advisor before making investment decisions."
        )
        
        ax.text(5, 3, disclaimer_text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7),
                wrap=True)
        
        # Footer
        ax.text(5, 1, f"Generated by {self.report_config['company_name']} | Confidential", 
                ha='center', va='center', fontsize=10, style='italic',
                color='gray')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_executive_summary(self, pdf, strategy: str, period: str):
        """Create executive summary page"""
        fig = plt.figure(figsize=(8.5, 11))
        
        # Get summary data
        summary_data = self._get_portfolio_summary(strategy, period)
        
        # Title
        fig.suptitle('Executive Summary', fontsize=20, fontweight='bold', y=0.95)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Key metrics boxes
        metrics = [
            ('Total Return', f"{summary_data['total_return']:.1f}%", 'return'),
            ('Annualized Return', f"{summary_data['annual_return']:.1f}%", 'annual'),
            ('Sharpe Ratio', f"{summary_data['sharpe_ratio']:.2f}", 'sharpe'),
            ('Max Drawdown', f"{summary_data['max_drawdown']:.1f}%", 'drawdown'),
            ('Total Value', f"${summary_data['total_value']:,.0f}", 'value'),
            ('Positions', f"{summary_data['total_positions']}", 'positions')
        ]
        
        for i, (label, value, metric_type) in enumerate(metrics[:4]):
            ax = fig.add_subplot(gs[0, i//2])
            
            # Color code based on performance
            if metric_type in ['return', 'annual', 'sharpe']:
                color = self.report_config['colors']['success'] if float(value.strip('%')) > 0 else self.report_config['colors']['danger']
            elif metric_type == 'drawdown':
                color = self.report_config['colors']['danger']
            else:
                color = self.report_config['colors']['primary']
            
            ax.text(0.5, 0.7, label, ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.text(0.5, 0.3, value, ha='center', va='center', fontsize=18, fontweight='bold',
                   color=color, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Add border
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='gray', 
                                         linewidth=1, transform=ax.transAxes))
        
        # Portfolio composition text
        ax_text = fig.add_subplot(gs[1:3, :])
        
        composition_text = f"""
PORTFOLIO COMPOSITION & PERFORMANCE

Your {strategy.replace('_', ' ').title()} strategy has {"outperformed" if summary_data['alpha'] > 0 else "underperformed"} 
the {self.report_config['benchmark']} benchmark by {abs(summary_data['alpha']):.1f}% over the {period} period.

Key Highlights:
• Portfolio Value: ${summary_data['total_value']:,.0f}
• Total Return: {summary_data['total_return']:.1f}% vs {summary_data['benchmark_return']:.1f}% for {self.report_config['benchmark']}
• Risk-Adjusted Performance: Sharpe ratio of {summary_data['sharpe_ratio']:.2f}
• Diversification: {summary_data['total_positions']} positions across {summary_data['sectors']} sectors

RISK MANAGEMENT
Maximum drawdown of {summary_data['max_drawdown']:.1f}% demonstrates {"strong" if summary_data['max_drawdown'] < 15 else "moderate" if summary_data['max_drawdown'] < 25 else "elevated"} 
risk control. Portfolio volatility of {summary_data['volatility']:.1f}% compares to {summary_data['benchmark_volatility']:.1f}% 
for the benchmark.

OUTLOOK
{"The strategy continues to demonstrate strong fundamentals-driven stock selection." if summary_data['alpha'] > 0 else "Recent underperformance may present opportunity as valuations remain attractive."}
        """
        
        ax_text.text(0.05, 0.95, composition_text.strip(), ha='left', va='top', fontsize=11,
                    transform=ax_text.transAxes, wrap=True)
        ax_text.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_performance_overview(self, pdf, strategy: str, period: str):
        """Create performance overview with charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Performance Overview', fontsize=16, fontweight='bold')
        
        # Get performance data
        performance_data = self._get_performance_data(strategy, period)
        
        if not performance_data.empty:
            # Cumulative return chart
            ax1.plot(performance_data.index, performance_data['portfolio_cumulative'], 
                    label='Portfolio', linewidth=2, color=self.report_config['colors']['primary'])
            ax1.plot(performance_data.index, performance_data['benchmark_cumulative'], 
                    label='S&P 500', linewidth=1.5, color=self.report_config['colors']['secondary'], alpha=0.8)
            ax1.set_title('Cumulative Returns', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Rolling returns
            rolling_portfolio = performance_data['portfolio_return'].rolling(window=30).mean() * 252
            rolling_benchmark = performance_data['benchmark_return'].rolling(window=30).mean() * 252
            
            ax2.plot(performance_data.index, rolling_portfolio, 
                    label='Portfolio (30d avg)', color=self.report_config['colors']['primary'])
            ax2.plot(performance_data.index, rolling_benchmark, 
                    label='S&P 500 (30d avg)', color=self.report_config['colors']['secondary'], alpha=0.8)
            ax2.set_title('Rolling 30-Day Annualized Returns', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Drawdown analysis
            portfolio_dd = self._calculate_drawdown(performance_data['portfolio_cumulative'])
            ax3.fill_between(performance_data.index, portfolio_dd, 0, 
                           color=self.report_config['colors']['danger'], alpha=0.3)
            ax3.plot(performance_data.index, portfolio_dd, 
                    color=self.report_config['colors']['danger'], linewidth=1.5)
            ax3.set_title('Portfolio Drawdown', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Monthly returns heatmap
            monthly_returns = performance_data['portfolio_return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns.index = monthly_returns.index.to_period('M')
            
            if len(monthly_returns) > 12:
                # Create pivot table for heatmap
                monthly_data = monthly_returns.to_frame()
                monthly_data['Year'] = monthly_returns.index.year
                monthly_data['Month'] = monthly_returns.index.month
                pivot_data = monthly_data.pivot(index='Year', columns='Month', values='portfolio_return')
                
                # Plot heatmap
                im = ax4.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
                ax4.set_title('Monthly Returns Heatmap', fontweight='bold')
                ax4.set_xticks(range(12))
                ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=8)
                ax4.set_yticks(range(len(pivot_data.index)))
                ax4.set_yticklabels(pivot_data.index, fontsize=8)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4)
                cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for\nmonthly heatmap', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.axis('off')
        
        else:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No performance data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_allocation_analysis(self, pdf, strategy: str):
        """Create portfolio allocation analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Portfolio Allocation Analysis', fontsize=16, fontweight='bold')
        
        # Get allocation data
        allocation_data = self._get_allocation_data(strategy)
        
        # Sector allocation pie chart
        if 'sector_allocation' in allocation_data and allocation_data['sector_allocation']:
            sectors = list(allocation_data['sector_allocation'].keys())
            sector_weights = list(allocation_data['sector_allocation'].values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))
            wedges, texts, autotexts = ax1.pie(sector_weights, labels=sectors, autopct='%1.1f%%',
                                              colors=colors, startangle=90)
            ax1.set_title('Sector Allocation', fontweight='bold')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax1.text(0.5, 0.5, 'No sector data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
        
        # Market cap allocation
        if 'market_cap_allocation' in allocation_data:
            market_caps = ['Small Cap', 'Mid Cap', 'Large Cap']
            market_cap_weights = [
                allocation_data['market_cap_allocation'].get('small_cap', 0),
                allocation_data['market_cap_allocation'].get('mid_cap', 0),
                allocation_data['market_cap_allocation'].get('large_cap', 0)
            ]
            
            bars = ax2.bar(market_caps, market_cap_weights, 
                          color=[self.report_config['colors']['primary'], 
                                self.report_config['colors']['secondary'],
                                self.report_config['colors']['success']])
            ax2.set_title('Market Cap Allocation', fontweight='bold')
            ax2.set_ylabel('Allocation %')
            
            # Add value labels on bars
            for bar, weight in zip(bars, market_cap_weights):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{weight:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No market cap data available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
        
        # Top 10 positions
        if 'top_positions' in allocation_data and allocation_data['top_positions']:
            positions_df = pd.DataFrame(allocation_data['top_positions'][:10])
            
            y_positions = range(len(positions_df))
            bars = ax3.barh(y_positions, positions_df['weight'], 
                           color=self.report_config['colors']['primary'], alpha=0.7)
            ax3.set_yticks(y_positions)
            ax3.set_yticklabels(positions_df['symbol'], fontsize=10)
            ax3.set_xlabel('Portfolio Weight (%)')
            ax3.set_title('Top 10 Holdings', fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            # Add weight labels
            for i, (bar, weight) in enumerate(zip(bars, positions_df['weight'])):
                ax3.text(weight + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{weight:.1f}%', ha='left', va='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No position data available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
        
        # Portfolio concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(strategy)
        
        ax4.text(0.1, 0.9, 'CONCENTRATION ANALYSIS', fontweight='bold', fontsize=12,
                transform=ax4.transAxes)
        
        metrics_text = f"""
Top 5 Holdings: {concentration_metrics['top_5_weight']:.1f}%
Top 10 Holdings: {concentration_metrics['top_10_weight']:.1f}%
Largest Position: {concentration_metrics['largest_position']:.1f}%

Herfindahl Index: {concentration_metrics['herfindahl_index']:.3f}
Effective # of Stocks: {concentration_metrics['effective_positions']:.0f}

Diversification Score: {concentration_metrics['diversification_score']}/10
        """
        
        ax4.text(0.1, 0.7, metrics_text.strip(), fontsize=10, transform=ax4.transAxes,
                verticalalignment='top')
        ax4.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_holdings_analysis(self, pdf, strategy: str):
        """Create detailed holdings analysis"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Holdings Analysis', fontsize=16, fontweight='bold')
        
        # Get holdings data
        holdings_data = self._get_detailed_holdings(strategy)
        
        if not holdings_data.empty:
            # Create table
            ax = fig.add_subplot(111)
            ax.axis('tight')
            ax.axis('off')
            
            # Select top 20 holdings for display
            top_holdings = holdings_data.head(20)
            
            # Prepare table data
            table_data = []
            for _, holding in top_holdings.iterrows():
                table_data.append([
                    holding['symbol'],
                    holding.get('company_name', 'N/A')[:25],  # Truncate long names
                    holding.get('sector', 'N/A'),
                    f"{holding['shares']:,.0f}",
                    f"${holding['avg_cost']:.2f}",
                    f"${holding['current_price']:.2f}",
                    f"${holding['market_value']:,.0f}",
                    f"{holding['weight']:.1f}%",
                    f"{holding['pnl_percent']:.1f}%"
                ])
            
            # Create table
            table = ax.table(cellText=table_data,
                           colLabels=['Symbol', 'Company', 'Sector', 'Shares', 'Avg Cost', 
                                    'Current Price', 'Market Value', 'Weight', 'P&L %'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.08, 0.2, 0.12, 0.08, 0.08, 0.08, 0.12, 0.07, 0.07])
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            
            # Color code P&L column
            for i in range(len(table_data)):
                pnl_value = float(table_data[i][-1].strip('%'))
                if pnl_value >= 0:
                    table[(i+1, 8)].set_facecolor('#d4edda')  # Light green
                else:
                    table[(i+1, 8)].set_facecolor('#f8d7da')  # Light red
            
            # Style header
            for j in range(len(table_data[0])):
                table[(0, j)].set_facecolor('#e9ecef')
                table[(0, j)].set_text_props(weight='bold')
            
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No holdings data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_risk_analysis(self, pdf, strategy: str, period: str):
        """Create comprehensive risk analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
        
        # Get risk metrics
        risk_data = self._get_risk_metrics(strategy, period)
        
        # Risk metrics summary
        ax1.text(0.1, 0.9, 'RISK METRICS', fontweight='bold', fontsize=14, transform=ax1.transAxes)
        
        risk_text = f"""
Volatility (Annual): {risk_data.get('volatility', 0):.1f}%
Sharpe Ratio: {risk_data.get('sharpe_ratio', 0):.2f}
Sortino Ratio: {risk_data.get('sortino_ratio', 0):.2f}
Maximum Drawdown: {risk_data.get('max_drawdown', 0):.1f}%
VaR (95%, 1-day): {risk_data.get('var_95', 0):.1f}%
Expected Shortfall: {risk_data.get('expected_shortfall', 0):.1f}%

Beta vs S&P 500: {risk_data.get('beta', 0):.2f}
Correlation vs S&P 500: {risk_data.get('correlation', 0):.2f}
Tracking Error: {risk_data.get('tracking_error', 0):.1f}%
Information Ratio: {risk_data.get('information_ratio', 0):.2f}
        """
        
        ax1.text(0.1, 0.8, risk_text.strip(), fontsize=10, transform=ax1.transAxes,
                verticalalignment='top')
        ax1.axis('off')
        
        # Return distribution histogram
        if 'daily_returns' in risk_data and len(risk_data['daily_returns']) > 0:
            returns = np.array(risk_data['daily_returns'])
            ax2.hist(returns, bins=50, alpha=0.7, color=self.report_config['colors']['primary'], density=True)
            ax2.axvline(returns.mean(), color=self.report_config['colors']['danger'], 
                       linestyle='--', label=f'Mean: {returns.mean():.2%}')
            ax2.axvline(np.percentile(returns, 5), color=self.report_config['colors']['warning'], 
                       linestyle='--', label=f'5th percentile: {np.percentile(returns, 5):.2%}')
            ax2.set_title('Daily Returns Distribution', fontweight='bold')
            ax2.set_xlabel('Daily Return')
            ax2.set_ylabel('Density')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        else:
            ax2.text(0.5, 0.5, 'No return data available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.axis('off')
        
        # Rolling volatility
        if 'rolling_volatility' in risk_data and len(risk_data['rolling_volatility']) > 0:
            vol_data = risk_data['rolling_volatility']
            ax3.plot(vol_data.index, vol_data.values, color=self.report_config['colors']['primary'], linewidth=1.5)
            ax3.fill_between(vol_data.index, vol_data.values, alpha=0.3, color=self.report_config['colors']['primary'])
            ax3.set_title('Rolling 30-Day Volatility', fontweight='bold')
            ax3.set_ylabel('Annualized Volatility')
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        else:
            ax3.text(0.5, 0.5, 'No volatility data available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.axis('off')
        
        # Risk/Return scatter
        if 'benchmark_data' in risk_data:
            portfolio_vol = risk_data.get('volatility', 0) / 100
            portfolio_ret = risk_data.get('annual_return', 0) / 100
            benchmark_vol = risk_data['benchmark_data'].get('volatility', 0) / 100
            benchmark_ret = risk_data['benchmark_data'].get('annual_return', 0) / 100
            
            ax4.scatter(portfolio_vol, portfolio_ret, s=100, color=self.report_config['colors']['primary'], 
                       label='Portfolio', alpha=0.8)
            ax4.scatter(benchmark_vol, benchmark_ret, s=100, color=self.report_config['colors']['secondary'], 
                       label='S&P 500', alpha=0.8)
            
            ax4.set_xlabel('Volatility (Annual)')
            ax4.set_ylabel('Return (Annual)')
            ax4.set_title('Risk-Return Profile', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        else:
            ax4.text(0.5, 0.5, 'No benchmark data available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_strategy_performance(self, pdf, strategy: str):
        """Create strategy-specific performance analysis"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # Get strategy performance data
        strategy_data = self._get_strategy_performance_data(strategy)
        
        # Create layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        # Strategy description
        ax1 = fig.add_subplot(gs[0, :])
        strategy_description = self._get_strategy_description(strategy)
        ax1.text(0.05, 0.95, 'STRATEGY OVERVIEW', fontweight='bold', fontsize=14, transform=ax1.transAxes)
        ax1.text(0.05, 0.75, strategy_description, fontsize=11, transform=ax1.transAxes, 
                verticalalignment='top', wrap=True)
        ax1.axis('off')
        
        # Performance attribution (if available)
        ax2 = fig.add_subplot(gs[1, 0])
        if 'sector_attribution' in strategy_data:
            sectors = list(strategy_data['sector_attribution'].keys())
            attributions = list(strategy_data['sector_attribution'].values())
            
            colors = ['green' if x >= 0 else 'red' for x in attributions]
            bars = ax2.barh(sectors, attributions, color=colors, alpha=0.7)
            ax2.set_title('Sector Attribution (bp)', fontweight='bold')
            ax2.set_xlabel('Attribution (basis points)')
            ax2.grid(axis='x', alpha=0.3)
            ax2.axvline(0, color='black', linewidth=0.8)
        else:
            ax2.text(0.5, 0.5, 'Attribution data\nnot available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.axis('off')
        
        # Factor exposure (if available)
        ax3 = fig.add_subplot(gs[1, 1])
        if 'factor_exposure' in strategy_data:
            factors = list(strategy_data['factor_exposure'].keys())
            exposures = list(strategy_data['factor_exposure'].values())
            
            colors = [self.report_config['colors']['primary'] if abs(x) > 0.5 else 'lightgray' for x in exposures]
            bars = ax3.bar(factors, exposures, color=colors, alpha=0.7)
            ax3.set_title('Factor Exposures', fontweight='bold')
            ax3.set_ylabel('Exposure')
            ax3.grid(axis='y', alpha=0.3)
            ax3.axhline(0, color='black', linewidth=0.8)
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Factor exposure\ndata not available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.axis('off')
        
        # Performance statistics table
        ax4 = fig.add_subplot(gs[2, :])
        perf_stats = strategy_data.get('performance_stats', {})
        
        if perf_stats:
            # Create performance comparison table
            periods = ['1M', '3M', '6M', '1Y', '2Y', '3Y', 'Inception']
            portfolio_returns = [perf_stats.get(f'portfolio_{p}', 0) for p in periods]
            benchmark_returns = [perf_stats.get(f'benchmark_{p}', 0) for p in periods]
            alpha_values = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
            
            table_data = []
            for i, period in enumerate(periods):
                table_data.append([
                    period,
                    f"{portfolio_returns[i]:.1f}%",
                    f"{benchmark_returns[i]:.1f}%",
                    f"{alpha_values[i]:+.1f}%"
                ])
            
            table = ax4.table(cellText=table_data,
                            colLabels=['Period', 'Portfolio', 'S&P 500', 'Alpha'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.15, 0.2, 0.2, 0.2])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)
            
            # Style header
            for j in range(4):
                table[(0, j)].set_facecolor('#e9ecef')
                table[(0, j)].set_text_props(weight='bold')
            
            # Color code alpha column
            for i in range(len(table_data)):
                alpha_val = alpha_values[i]
                if alpha_val >= 0:
                    table[(i+1, 3)].set_facecolor('#d4edda')  # Light green
                else:
                    table[(i+1, 3)].set_facecolor('#f8d7da')  # Light red
        
        ax4.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_market_commentary(self, pdf, period: str):
        """Create market commentary and outlook"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Market Commentary & Outlook', fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Get current date for commentary
        current_date = datetime.now().strftime("%B %Y")
        
        commentary_text = f"""
MARKET ENVIRONMENT - {current_date.upper()}

EXECUTIVE SUMMARY
The {period} reporting period has been characterized by evolving market dynamics driven by macroeconomic factors, 
corporate earnings trends, and evolving investor sentiment. Our quantitative strategies have navigated these conditions 
through disciplined fundamental analysis and systematic risk management.

KEY MARKET THEMES

Economic Environment
• Interest rate environment continues to influence equity valuations across market segments
• Corporate earnings growth has shown resilience despite economic headwinds
• Sector rotation has created opportunities for active stock selection

Market Dynamics  
• Value vs Growth rotation has impacted relative performance across strategies
• Small and mid-cap stocks have shown increased dispersion, benefiting active management
• Quality factors have remained important in stock selection processes

PORTFOLIO POSITIONING

Current Positioning
Our systematic approach has maintained disciplined exposure to high-quality companies with strong fundamentals:
• Emphasis on companies with sustainable competitive advantages
• Focus on reasonable valuations relative to growth prospects
• Diversification across sectors while avoiding excessive concentration risk

Risk Management
• Position sizing based on conviction and risk metrics
• Sector allocation limits to prevent overconcentration
• Ongoing monitoring of portfolio-level risk characteristics

OUTLOOK

Market Outlook
We maintain a constructive long-term view while remaining vigilant about near-term risks:
• Continued focus on fundamental stock selection
• Opportunistic positioning as market dislocations create value
• Emphasis on quality and sustainable business models

Strategy Evolution
Our quantitative models continue to evolve with market conditions:
• Regular model updates based on changing market dynamics
• Integration of ESG factors where material to investment outcomes
• Enhanced risk management through advanced analytics

CONCLUSION
The systematic, research-driven approach continues to be well-positioned for long-term value creation. 
While short-term market volatility may persist, our focus on fundamental analysis and disciplined 
risk management provides a strong foundation for navigating various market environments.

This commentary reflects our current market views and is subject to change based on evolving conditions.
Past performance does not guarantee future results.
        """
        
        ax.text(0.05, 0.95, commentary_text.strip(), fontsize=11, transform=ax.transAxes,
               verticalalignment='top', wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_appendix(self, pdf, strategy: str):
        """Create appendix with methodology and disclosures"""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Appendix - Methodology & Disclosures', fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        appendix_text = """
METHODOLOGY

Performance Calculation
• All returns are calculated using time-weighted return methodology
• Returns are presented gross of fees unless otherwise specified
• Benchmark returns include reinvestment of dividends
• Performance inception date represents first full month of strategy implementation

Risk Metrics Definitions
• Volatility: Annualized standard deviation of daily returns
• Sharpe Ratio: Excess return over risk-free rate divided by volatility
• Maximum Drawdown: Largest peak-to-trough decline during the period
• Beta: Sensitivity of portfolio returns to benchmark returns
• Alpha: Excess return versus benchmark after adjusting for beta

Portfolio Construction
• Stock selection based on proprietary fundamental scoring model
• Portfolio optimization considers expected returns, risk, and transaction costs
• Regular rebalancing to maintain target allocations
• Position limits to control concentration risk

Data Sources
• Financial data sourced from leading market data providers
• Fundamental data undergoes quality control and standardization
• Real-time pricing for portfolio valuation and risk monitoring

IMPORTANT DISCLOSURES

General Disclaimers
This report is provided for informational purposes only and should not be considered as investment advice 
or a recommendation to buy or sell any securities. Past performance is not indicative of future results.

Risk Considerations
• All investments carry risk of loss of principal
• Performance may be volatile and subject to significant drawdowns  
• Strategy performance may deviate substantially from benchmark returns
• Concentration in certain sectors or securities may increase portfolio risk

Performance Disclosure
• Results shown are for a model portfolio and may differ from actual client accounts
• Performance data includes periods of both favorable and unfavorable market conditions
• Individual client returns may vary due to timing of contributions and withdrawals
• Transaction costs and management fees will reduce returns

Forward-Looking Statements
This report may contain forward-looking statements based on current expectations and assumptions. 
Actual results may differ materially from those projected. No assurance can be given that any 
investment objectives will be achieved.

Additional Information
For more detailed information about methodology, risks, and performance, please refer to the 
strategy's offering documents and consult with your investment advisor.

Report prepared by ACIS Trading Platform quantitative research team.
For questions or additional information, please contact your relationship manager.

© 2024 ACIS Trading Platform. All rights reserved. This report is confidential and proprietary.
        """
        
        ax.text(0.05, 0.95, appendix_text.strip(), fontsize=9, transform=ax.transAxes,
               verticalalignment='top')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # Helper methods for data retrieval
    def _get_portfolio_summary(self, strategy: str, period: str) -> dict:
        """Get portfolio summary metrics"""
        # This would query the database for actual portfolio data
        return {
            'total_return': 12.5,
            'annual_return': 15.2,
            'sharpe_ratio': 1.18,
            'max_drawdown': -8.3,
            'total_value': 1250000,
            'total_positions': 45,
            'sectors': 8,
            'alpha': 3.8,
            'benchmark_return': 8.7,
            'volatility': 18.5,
            'benchmark_volatility': 16.2
        }
    
    def _get_performance_data(self, strategy: str, period: str) -> pd.DataFrame:
        """Get historical performance data"""
        # Generate sample data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        np.random.seed(42)
        
        portfolio_returns = np.random.normal(0.0008, 0.015, len(dates))
        benchmark_returns = np.random.normal(0.0005, 0.012, len(dates))
        
        portfolio_cumulative = (1 + pd.Series(portfolio_returns)).cumprod() - 1
        benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod() - 1
        
        return pd.DataFrame({
            'portfolio_return': portfolio_returns,
            'benchmark_return': benchmark_returns,
            'portfolio_cumulative': portfolio_cumulative,
            'benchmark_cumulative': benchmark_cumulative
        }, index=dates)
    
    def _get_allocation_data(self, strategy: str) -> dict:
        """Get portfolio allocation data"""
        return {
            'sector_allocation': {
                'Technology': 22.5,
                'Financial Services': 18.3,
                'Healthcare': 15.2,
                'Industrial': 12.8,
                'Consumer Discretionary': 10.1,
                'Materials': 8.7,
                'Energy': 6.2,
                'Utilities': 4.1,
                'Real Estate': 2.1
            },
            'market_cap_allocation': {
                'small_cap': 35.0,
                'mid_cap': 40.0,
                'large_cap': 25.0
            },
            'top_positions': [
                {'symbol': 'AAPL', 'weight': 3.2, 'market_value': 40000},
                {'symbol': 'MSFT', 'weight': 2.8, 'market_value': 35000},
                {'symbol': 'GOOGL', 'weight': 2.5, 'market_value': 31250},
                {'symbol': 'AMZN', 'weight': 2.3, 'market_value': 28750},
                {'symbol': 'NVDA', 'weight': 2.1, 'market_value': 26250}
            ]
        }
    
    def _get_detailed_holdings(self, strategy: str) -> pd.DataFrame:
        """Get detailed holdings data"""
        # Generate sample holdings data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JNJ', 'V']
        
        holdings_data = []
        for i, symbol in enumerate(symbols):
            holdings_data.append({
                'symbol': symbol,
                'company_name': f'{symbol} Inc.',
                'sector': 'Technology' if i < 5 else 'Other',
                'shares': (1000 - i * 50),
                'avg_cost': 150.0 + i * 20,
                'current_price': 160.0 + i * 25,
                'market_value': (1000 - i * 50) * (160.0 + i * 25),
                'weight': max(3.5 - i * 0.3, 0.5),
                'pnl_percent': (160.0 + i * 25) / (150.0 + i * 20) * 100 - 100
            })
        
        return pd.DataFrame(holdings_data)
    
    def _get_risk_metrics(self, strategy: str, period: str) -> dict:
        """Get comprehensive risk metrics"""
        # Generate sample risk data
        np.random.seed(42)
        daily_returns = np.random.normal(0.0008, 0.015, 252)
        
        return {
            'volatility': 18.5,
            'sharpe_ratio': 1.18,
            'sortino_ratio': 1.45,
            'max_drawdown': -8.3,
            'var_95': -2.1,
            'expected_shortfall': -3.2,
            'beta': 1.05,
            'correlation': 0.85,
            'tracking_error': 4.2,
            'information_ratio': 0.95,
            'daily_returns': daily_returns,
            'rolling_volatility': pd.Series(
                np.random.uniform(0.12, 0.25, 100),
                index=pd.date_range(end=datetime.now(), periods=100, freq='D')
            ),
            'benchmark_data': {
                'volatility': 16.2,
                'annual_return': 8.7
            }
        }
    
    def _get_strategy_performance_data(self, strategy: str) -> dict:
        """Get strategy-specific performance data"""
        return {
            'sector_attribution': {
                'Technology': 45,
                'Healthcare': 23,
                'Financial': -12,
                'Industrial': 18,
                'Consumer': -8
            },
            'factor_exposure': {
                'Value': 0.3,
                'Growth': -0.2,
                'Quality': 0.8,
                'Momentum': 0.1,
                'Size': -0.5
            },
            'performance_stats': {
                'portfolio_1M': 2.1, 'benchmark_1M': 1.8,
                'portfolio_3M': 6.5, 'benchmark_3M': 5.2,
                'portfolio_6M': 8.3, 'benchmark_6M': 6.9,
                'portfolio_1Y': 15.2, 'benchmark_1Y': 11.4,
                'portfolio_2Y': 28.7, 'benchmark_2Y': 22.1,
                'portfolio_3Y': 45.2, 'benchmark_3Y': 35.8,
                'portfolio_Inception': 65.4, 'benchmark_Inception': 48.2
            }
        }
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get strategy description"""
        descriptions = {
            'small_cap_value': 'Focuses on undervalued small-cap companies with strong fundamentals and attractive valuations.',
            'mid_cap_growth': 'Targets mid-cap companies with sustainable growth prospects and expanding market opportunities.',
            'large_cap_momentum': 'Invests in large-cap stocks showing strong price and earnings momentum.',
            'dividend_focus': 'Emphasizes companies with consistent dividend payments and sustainable payout ratios.',
            'all': 'Multi-strategy approach combining value, growth, momentum, and dividend strategies across market caps.'
        }
        return descriptions.get(strategy, 'Quantitative equity strategy focused on systematic stock selection.')
    
    def _calculate_concentration_metrics(self, strategy: str) -> dict:
        """Calculate portfolio concentration metrics"""
        return {
            'top_5_weight': 15.2,
            'top_10_weight': 25.8,
            'largest_position': 3.2,
            'herfindahl_index': 0.045,
            'effective_positions': 22.2,
            'diversification_score': 8
        }
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown

def main():
    """Generate sample client report"""
    print("ACIS Client Report Generator")
    print("Generating comprehensive portfolio report...")
    
    generator = ClientReportGenerator()
    
    # Generate report for sample client
    report_path = generator.generate_client_report(
        client_name="John Smith",
        strategy="small_cap_value",
        report_period="1Y"
    )
    
    print(f"\nClient report generated successfully!")
    print(f"File: {report_path}")
    print(f"Size: {os.path.getsize(report_path) / 1024:.1f} KB")

if __name__ == "__main__":
    main()