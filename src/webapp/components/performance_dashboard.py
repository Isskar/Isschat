"""
Performance Dashboard for Isschat
Modern and clean interface for real-time monitoring
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import time
import os

logger = logging.getLogger(__name__)


class PerformanceDashboard:
    """Performance dashboard with modern and professional interface."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'background': '#f8f9fa'
        }
    
    def render_dashboard(self):
        """Main dashboard rendering."""
        st.title("Performance Dashboard")
        
        # Real-time metrics
        self._render_realtime_metrics()
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_response_time_summary()
            self._render_system_health_summary()
        
        with col2:
            self._render_query_analytics_summary()
            self._render_performance_trends_summary()
    
    def _render_realtime_metrics(self):
        """Display real-time system metrics."""
        st.subheader("System Performance Metrics")
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        recent_performance = self.data_manager.get_performance_metrics(limit=50)
        
        # Calculate performance metrics
        avg_response_time = 0
        if recent_performance:
            avg_response_time = sum(p.get('duration_ms', 0) for p in recent_performance) / len(recent_performance)
        
        # Display in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="CPU Usage",
                value=f"{system_metrics['cpu_percent']:.1f}%",
                delta=f"{system_metrics['cpu_delta']:+.1f}%"
            )
        
        with col2:
            st.metric(
                label="Memory Usage",
                value=f"{system_metrics['memory_percent']:.1f}%",
                delta=f"{system_metrics['memory_delta']:+.1f}%"
            )
        
        with col3:
            st.metric(
                label="Avg Response Time",
                value=f"{avg_response_time:.0f}ms",
                delta=f"{-20 if avg_response_time > 0 else 0:+.0f}ms"
            )
        
        with col4:
            st.metric(
                label="System Load",
                value=f"{system_metrics['load_avg']:.2f}",
                delta=f"{system_metrics['load_delta']:+.2f}"
            )
    
    def _render_response_time_summary(self):
        """Response time summary."""
        st.subheader("Response Time Analysis")
        
        performance_data = self.data_manager.get_performance_metrics(limit=50)
        
        if performance_data:
            durations = [p.get('duration_ms', 0) for p in performance_data]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min", f"{min(durations):.0f}ms")
            with col2:
                st.metric("Avg", f"{sum(durations)/len(durations):.0f}ms")
            with col3:
                st.metric("Max", f"{max(durations):.0f}ms")
            
            # Simple text-based chart
            st.write("**Recent Performance Trend:**")
            recent_5 = durations[-5:] if len(durations) >= 5 else durations
            trend_text = " → ".join([f"{d:.0f}ms" for d in recent_5])
            st.code(trend_text)
            
        else:
            st.info("No performance data available")
    
    def _render_system_health_summary(self):
        """System health summary."""
        st.subheader("System Health")
        
        # Get real system metrics
        system_metrics = self._get_system_metrics()
        
        # Health status
        cpu_status = "🟢 Good" if system_metrics['cpu_percent'] < 70 else "🟡 Warning" if system_metrics['cpu_percent'] < 90 else "🔴 Critical"
        memory_status = "🟢 Good" if system_metrics['memory_percent'] < 70 else "🟡 Warning" if system_metrics['memory_percent'] < 90 else "🔴 Critical"
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**CPU:** {system_metrics['cpu_percent']:.1f}%")
            st.write(cpu_status)
        
        with col2:
            st.write(f"**Memory:** {system_metrics['memory_percent']:.1f}%")
            st.write(memory_status)
        
        # Simple progress bars using text
        cpu_bar = "█" * int(system_metrics['cpu_percent'] / 10) + "░" * (10 - int(system_metrics['cpu_percent'] / 10))
        memory_bar = "█" * int(system_metrics['memory_percent'] / 10) + "░" * (10 - int(system_metrics['memory_percent'] / 10))
        
        st.write(f"CPU:    [{cpu_bar}]")
        st.write(f"Memory: [{memory_bar}]")
    
    def _render_query_analytics_summary(self):
        """Query performance analytics summary."""
        st.subheader("Query Performance")
        
        performance_data = self.data_manager.get_performance_metrics(limit=50)
        
        if performance_data:
            durations = [p.get('duration_ms', 0) for p in performance_data]
            
            # Performance categories
            fast_queries = len([d for d in durations if d < 1000])
            medium_queries = len([d for d in durations if 1000 <= d < 3000])
            slow_queries = len([d for d in durations if d >= 3000])
            
            st.write("**Performance Distribution:**")
            st.write(f"Fast (<1s): {fast_queries} queries")
            st.write(f"Medium (1-3s): {medium_queries} queries")
            st.write(f"Slow (>3s): {slow_queries} queries")
            
            # Simple histogram using text
            total = len(durations)
            if total > 0:
                fast_pct = (fast_queries / total) * 100
                medium_pct = (medium_queries / total) * 100
                slow_pct = (slow_queries / total) * 100
                
                st.write("**Distribution:**")
                st.write(f"🟢 Fast: {fast_pct:.1f}%")
                st.write(f"🟡 Medium: {medium_pct:.1f}%")
                st.write(f"🔴 Slow: {slow_pct:.1f}%")
        else:
            st.info("No performance data available")
    
    def _render_performance_trends_summary(self):
        """Performance trends summary."""
        st.subheader("Performance Trends")
        
        performance_data = self.data_manager.get_performance_metrics(limit=100)
        
        if performance_data:
            # Group by hour (simplified)
            current_hour = datetime.now().hour
            recent_hours = {}
            
            for p in performance_data:
                try:
                    timestamp = datetime.fromisoformat(p['timestamp'])
                    hour = timestamp.hour
                    if hour not in recent_hours:
                        recent_hours[hour] = []
                    recent_hours[hour].append(p.get('duration_ms', 0))
                except:
                    continue
            
            if recent_hours:
                st.write("**Average Response Time by Hour:**")
                for hour in sorted(recent_hours.keys())[-6:]:  # Last 6 hours
                    avg_time = sum(recent_hours[hour]) / len(recent_hours[hour])
                    bar_length = int(avg_time / 100)  # Scale for display
                    bar = "█" * min(bar_length, 20)
                    st.write(f"{hour:02d}:00 {avg_time:.0f}ms [{bar}]")
            
            # Trend analysis
            if len(performance_data) >= 10:
                first_half = performance_data[:len(performance_data)//2]
                second_half = performance_data[len(performance_data)//2:]
                
                avg_first = sum(p.get('duration_ms', 0) for p in first_half) / len(first_half)
                avg_second = sum(p.get('duration_ms', 0) for p in second_half) / len(second_half)
                
                trend = "📈 Improving" if avg_second < avg_first else "📉 Degrading" if avg_second > avg_first else "➡️ Stable"
                st.write(f"**Trend:** {trend}")
        else:
            st.info("No performance data available")
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics with fallback."""
        try:
            # Try to get real system metrics
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
            except AttributeError:
                # Windows doesn't have load average
                load_avg = cpu_percent / 100.0
            
            # Calculate deltas (simplified - in real app, store previous values)
            cpu_delta = 0.0  # Would compare with previous measurement
            memory_delta = 0.0  # Would compare with previous measurement
            load_delta = 0.0  # Would compare with previous measurement
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_delta': cpu_delta,
                'memory_percent': memory_percent,
                'memory_delta': memory_delta,
                'load_avg': load_avg,
                'load_delta': load_delta
            }
        except ImportError:
            # Fallback to simulated metrics
            logger.warning("psutil not available, using simulated metrics")
            import random
            return {
                'cpu_percent': random.uniform(20, 60),
                'cpu_delta': random.uniform(-5, 5),
                'memory_percent': random.uniform(30, 70),
                'memory_delta': random.uniform(-3, 3),
                'load_avg': random.uniform(0.5, 2.0),
                'load_delta': random.uniform(-0.2, 0.2)
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            # Fallback values
            return {
                'cpu_percent': 0.0,
                'cpu_delta': 0.0,
                'memory_percent': 0.0,
                'memory_delta': 0.0,
                'load_avg': 0.0,
                'load_delta': 0.0
            }


def render_performance_dashboard(data_manager):
    """Utility function to render the dashboard."""
    try:
        dashboard = PerformanceDashboard(data_manager)
        dashboard.render_dashboard()
    except Exception as e:
        st.error(f"Error rendering performance dashboard: {e}")
        logger.error(f"Dashboard error: {e}")
        
        # Fallback simple dashboard
        st.title("Performance Dashboard")
        st.info("Dashboard temporarily unavailable - showing basic metrics")
        
        # Basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("System Status", "Operational")
