import pandas as pd
import numpy as np
import streamlit as st
import json
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class PerformanceTracker:
    """Suit les performances du système RAG et fournit des métriques d'analyse"""
    
    def __init__(self, log_path="./logs/performance"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(log_path, f"perf_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
    
    def log_performance(self, metrics):
        """Enregistre les métriques de performance d'une requête"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def track_query(self, question, retrieval_time, generation_time, num_docs_retrieved, 
                   embedding_time=None, total_time=None, memory_usage=None):
        """Enregistre les métriques de performance d'une requête complète"""
        metrics = {
            "question": question,
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": generation_time,
            "num_docs_retrieved": num_docs_retrieved,
            "total_time_ms": total_time or (retrieval_time + generation_time),
            "embedding_time_ms": embedding_time,
            "memory_usage_mb": memory_usage
        }
        
        self.log_performance(metrics)
        return metrics
    
    def get_performance_logs(self, days=7):
        """Récupère les logs de performance des n derniers jours"""
        logs = []
        
        # Trouver tous les fichiers de log dans la période spécifiée
        for filename in os.listdir(self.log_path):
            if filename.startswith("perf_log_") and filename.endswith(".jsonl"):
                file_path = os.path.join(self.log_path, filename)
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_date = datetime.fromisoformat(log_entry["timestamp"])
                            days_ago = (datetime.now() - log_date).days
                            if days_ago <= days:
                                logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue
        
        return logs
    
    def analyze_performance(self, logs=None):
        """Analyse les métriques de performance"""
        if logs is None:
            logs = self.get_performance_logs()
        
        if not logs:
            return {"message": "Aucune donnée disponible pour l'analyse"}
        
        # Convertir en DataFrame pour faciliter l'analyse
        df = pd.DataFrame(logs)
        
        # Ajouter la date/heure pour l'analyse temporelle
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        
        # Métriques globales
        metrics = {
            "total_queries": len(df),
            "avg_total_time_ms": df['total_time_ms'].mean(),
            "avg_retrieval_time_ms": df['retrieval_time_ms'].mean(),
            "avg_generation_time_ms": df['generation_time_ms'].mean(),
            "avg_docs_retrieved": df['num_docs_retrieved'].mean(),
            "p95_total_time_ms": df['total_time_ms'].quantile(0.95),
            "max_total_time_ms": df['total_time_ms'].max(),
            "min_total_time_ms": df['total_time_ms'].min()
        }
        
        # Tendances temporelles (par jour)
        daily_metrics = df.groupby('date').agg({
            'total_time_ms': 'mean',
            'retrieval_time_ms': 'mean',
            'generation_time_ms': 'mean',
            'timestamp': 'count'  # Nombre de requêtes par jour
        }).rename(columns={'timestamp': 'query_count'})
        
        # Tendances par heure
        hourly_metrics = df.groupby('hour').agg({
            'total_time_ms': 'mean',
            'timestamp': 'count'  # Nombre de requêtes par heure
        }).rename(columns={'timestamp': 'query_count'})
        
        return {
            "metrics": metrics,
            "daily_metrics": daily_metrics.reset_index().to_dict('records'),
            "hourly_metrics": hourly_metrics.reset_index().to_dict('records')
        }
    
    def render_performance_dashboard(self):
        """Display the performance dashboard in Streamlit"""
        st.title("Performance Tracking")
        
        # Period selection
        days = st.slider("Analysis period (days)", 1, 30, 7, key="perf_days")
        logs = self.get_performance_logs(days)
        
        if not logs:
            st.warning("No data available for the selected period")
            return
        
        # Analysis
        analysis = self.analyze_performance(logs)
        
        # Display main metrics
        metrics = analysis["metrics"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average response time", f"{metrics['avg_total_time_ms']:.0f} ms")
        with col2:
            st.metric("Average retrieval time", f"{metrics['avg_retrieval_time_ms']:.0f} ms")
        with col3:
            st.metric("Average generation time", f"{metrics['avg_generation_time_ms']:.0f} ms")
        
        # Time evolution chart
        st.subheader("Response Time Evolution")
        
        # Convert data for the chart
        daily_data = pd.DataFrame(analysis["daily_metrics"])
        if not daily_data.empty:
            daily_data['date'] = pd.to_datetime(daily_data['date'])
            daily_data = daily_data.sort_values('date')
            
            # Create chart
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Response times
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Time (ms)', color='tab:blue')
            ax1.plot(daily_data['date'], daily_data['total_time_ms'], 'b-', label='Total time')
            ax1.plot(daily_data['date'], daily_data['retrieval_time_ms'], 'g--', label='Retrieval time')
            ax1.plot(daily_data['date'], daily_data['generation_time_ms'], 'r--', label='Generation time')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.legend(loc='upper left')
            
            # Number of queries
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of queries', color='tab:orange')
            ax2.bar(daily_data['date'], daily_data['query_count'], alpha=0.3, color='tab:orange', label='Queries')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.legend(loc='upper right')
            
            fig.tight_layout()
            st.pyplot(fig)
        
        # Distribution by hour
        st.subheader("Performance distribution by hour")
        hourly_data = pd.DataFrame(analysis["hourly_metrics"])
        if not hourly_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='hour', y='total_time_ms', data=hourly_data, ax=ax, color='skyblue')
            ax.set_xlabel('Hour of day')
            ax.set_ylabel('Average response time (ms)')
            ax.set_title('Average response time by hour')
            st.pyplot(fig)


# Fonction pour intégrer le tracker de performance dans l'application principale
def integrate_performance_tracker(help_desk):
    """Intègre le tracker de performance au help_desk"""
    tracker = PerformanceTracker()
    
    # Fonction wrapper pour ask_question qui mesure les performances
    original_ask = help_desk.ask_question
    
    def ask_with_performance_tracking(question, verbose=False):
        # Mesurer le temps total
        start_time = time.time()
        
        # Mesurer le temps d'embedding (approximation)
        embedding_start = time.time()
        # Cette partie dépend de l'implémentation exacte de votre help_desk
        # Nous supposons que l'embedding est fait avant la recherche
        embedding_time = (time.time() - embedding_start) * 1000  # en ms
        
        # Mesurer le temps de recherche
        retrieval_start = time.time()
        # Cette partie dépend de l'implémentation exacte de votre help_desk
        # Nous supposons que la recherche est faite avant la génération
        retrieval_time = (time.time() - retrieval_start) * 1000  # en ms
        
        # Appeler la fonction originale pour obtenir la réponse
        answer, sources = original_ask(question, verbose)
        
        # Calculer le temps total et le temps de génération
        total_time = (time.time() - start_time) * 1000  # en ms
        generation_time = total_time - retrieval_time - embedding_time
        
        # Enregistrer les métriques
        tracker.track_query(
            question=question,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_docs_retrieved=len(sources) if sources else 0,
            embedding_time=embedding_time,
            total_time=total_time
        )
        
        return answer, sources
    
    # Remplacer la méthode originale
    help_desk.ask_question = ask_with_performance_tracking
    
    # Ajouter le tracker comme attribut
    help_desk.performance_tracker = tracker
    
    return help_desk
