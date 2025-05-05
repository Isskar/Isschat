import pandas as pd
import json
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st

class ConversationAnalyzer:
    """Analyse les conversations et fournit des insights sur les interactions utilisateur-chatbot"""
    
    def __init__(self, log_path="./logs/conversations"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_log_file = os.path.join(log_path, f"conv_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
    
    def log_interaction(self, user_id, question, answer, sources, response_time, feedback=None):
        """Enregistre une interaction dans le fichier de log"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer_length": len(answer),
            "sources_count": len(sources) if sources else 0,
            "response_time_ms": response_time,
            "feedback": feedback
        }
        
        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_recent_logs(self, days=7):
        """Récupère les logs des n derniers jours"""
        logs = []
        
        # Trouver tous les fichiers de log dans la période spécifiée
        for filename in os.listdir(self.log_path):
            if filename.startswith("conv_log_") and filename.endswith(".jsonl"):
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
    
    def analyze_questions(self, logs=None):
        """Analyse les questions posées"""
        if logs is None:
            logs = self.get_recent_logs()
        
        if not logs:
            return {"message": "Aucune donnée disponible pour l'analyse"}
        
        # Extraire les questions
        questions = [log["question"] for log in logs]
        
        # Analyse simple des mots-clés
        words = []
        for q in questions:
            words.extend([w.lower() for w in q.split() if len(w) > 3])
        
        common_words = Counter(words).most_common(10)
        
        # Analyse des heures de la journée
        hours = [datetime.fromisoformat(log["timestamp"]).hour for log in logs]
        hour_counts = Counter(hours)
        
        # Temps de réponse moyen
        avg_response_time = sum(log.get("response_time_ms", 0) for log in logs) / len(logs)
        
        return {
            "total_questions": len(questions),
            "common_words": common_words,
            "hour_distribution": dict(hour_counts),
            "avg_response_time_ms": avg_response_time
        }
    
    def render_analysis_dashboard(self):
        """Affiche le tableau de bord d'analyse dans Streamlit"""
        st.title("Analyse Conversationnelle")
        
        # Sélection de la période
        days = st.slider("Période d'analyse (jours)", 1, 30, 7)
        logs = self.get_recent_logs(days)
        
        if not logs:
            st.warning("Aucune donnée disponible pour la période sélectionnée")
            return
        
        # Analyse
        analysis = self.analyze_questions(logs)
        
        # Affichage des métriques
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total des questions", analysis["total_questions"])
        with col2:
            st.metric("Temps de réponse moyen", f"{analysis['avg_response_time_ms']:.0f} ms")
        
        # Graphique de distribution horaire
        st.subheader("Distribution des questions par heure")
        hours = list(range(24))
        counts = [analysis["hour_distribution"].get(hour, 0) for hour in hours]
        
        hour_df = pd.DataFrame({
            "Heure": hours,
            "Nombre de questions": counts
        })
        st.bar_chart(hour_df.set_index("Heure"))
        
        # Mots les plus fréquents
        st.subheader("Mots-clés les plus fréquents")
        if analysis["common_words"]:
            keywords_df = pd.DataFrame(analysis["common_words"], columns=["Mot", "Fréquence"])
            st.dataframe(keywords_df)
        else:
            st.info("Pas assez de données pour l'analyse des mots-clés")


# Fonction pour intégrer l'analyseur dans l'application principale
def integrate_conversation_analyzer(help_desk, user_id):
    """Intègre l'analyseur de conversation au help_desk"""
    analyzer = ConversationAnalyzer()
    
    # Fonction wrapper pour ask_question qui enregistre les interactions
    original_ask = help_desk.ask_question
    
    def ask_with_logging(question, verbose=False):
        start_time = datetime.now()
        answer, sources = original_ask(question, verbose)
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000  # en millisecondes
        
        # Enregistrer l'interaction
        analyzer.log_interaction(
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources,
            response_time=response_time
        )
        
        return answer, sources
    
    # Remplacer la méthode originale
    help_desk.ask_question = ask_with_logging
    
    # Ajouter l'analyseur comme attribut
    help_desk.analyzer = analyzer
    
    return help_desk
