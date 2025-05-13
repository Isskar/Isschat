import re
import json
import os


class QueryReformulator:
    """Reformule automatiquement les requêtes pour améliorer la qualité des réponses"""

    def __init__(self, cache_path="./cache"):
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
        self.reformulation_patterns_file = os.path.join(cache_path, "reformulation_patterns.json")
        self.patterns = self._load_patterns()

    def _load_patterns(self):
        """Charge les patterns de reformulation depuis le cache ou crée des patterns par défaut"""
        if os.path.exists(self.reformulation_patterns_file):
            try:
                with open(self.reformulation_patterns_file, "r") as f:
                    return json.load(f)
            except:
                return self._create_default_patterns()
        else:
            return self._create_default_patterns()

    def _create_default_patterns(self):
        """Crée des patterns de reformulation par défaut"""
        default_patterns = {
            "expansion": [
                {"pattern": r"\b(\w+)\b", "replacement": r"contexte \1 détaillé"},
                {
                    "pattern": r"comment (.+)",
                    "replacement": r"explique en détail comment \1",
                },
                {
                    "pattern": r"qu'est-ce que (.+)",
                    "replacement": r"définis et explique en détail \1",
                },
                {
                    "pattern": r"pourquoi (.+)",
                    "replacement": r"explique les raisons pour lesquelles \1",
                },
            ],
            "precision": [
                {
                    "pattern": r"information sur (.+)",
                    "replacement": r"informations spécifiques et détaillées sur \1",
                },
                {
                    "pattern": r"parle de (.+)",
                    "replacement": r"explique en détail \1 avec des exemples concrets",
                },
                {
                    "pattern": r"je veux savoir (.+)",
                    "replacement": r"fournir des informations détaillées sur \1",
                },
            ],
            "context": [
                {
                    "pattern": r"(.+) projet (.+)",
                    "replacement": r"\1 projet \2 dans le contexte de notre entreprise",
                },
                {
                    "pattern": r"(.+) processus (.+)",
                    "replacement": r"\1 processus \2 selon notre documentation interne",
                },
            ],
        }

        with open(self.reformulation_patterns_file, "w") as f:
            json.dump(default_patterns, f)

        return default_patterns

    def _save_patterns(self):
        """Sauvegarde les patterns de reformulation dans le cache"""
        with open(self.reformulation_patterns_file, "w") as f:
            json.dump(self.patterns, f)

    def add_pattern(self, category, pattern, replacement):
        """Ajoute un nouveau pattern de reformulation"""
        if category not in self.patterns:
            self.patterns[category] = []

        self.patterns[category].append({"pattern": pattern, "replacement": replacement})

        self._save_patterns()

    def reformulate_query(self, query, strategy="auto"):
        """Reformule une requête selon la stratégie spécifiée"""
        original_query = query

        if strategy == "auto":
            # Déterminer automatiquement la meilleure stratégie
            if len(query.split()) <= 3:
                strategy = "expansion"  # Requête courte -> expansion
            elif any(word in query.lower() for word in ["comment", "pourquoi", "quoi", "qui"]):
                strategy = "precision"  # Question -> précision
            else:
                strategy = "context"  # Par défaut -> contexte

        # Appliquer les patterns de la stratégie choisie
        if strategy in self.patterns:
            for pattern_obj in self.patterns[strategy]:
                pattern = pattern_obj["pattern"]
                replacement = pattern_obj["replacement"]
                query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

        # Éviter les reformulations identiques ou trop longues
        if query == original_query or len(query) > 3 * len(original_query):
            return original_query

        return query

    def suggest_reformulations(self, query, max_suggestions=3):
        """Suggère plusieurs reformulations possibles"""
        suggestions = []

        # Appliquer chaque stratégie
        for strategy in self.patterns:
            reformulation = self.reformulate_query(query, strategy)
            if reformulation != query and reformulation not in suggestions:
                suggestions.append(reformulation)

        # Limiter le nombre de suggestions
        return suggestions[:max_suggestions]

    def render_reformulation_widget(self, st, query, callback=None):
        """Affiche un widget de reformulation dans Streamlit"""
        suggestions = self.suggest_reformulations(query)

        if not suggestions:
            return query

        st.write("---")
        st.write("### Reformulations suggérées")
        st.write("Votre requête pourrait être plus précise. Essayez ces reformulations:")

        selected_query = query  # Par défaut, on garde la requête originale

        for i, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"reform_{i}"):
                selected_query = suggestion
                if callback:
                    callback(suggestion)

        return selected_query

    def render_admin_interface(self, st):
        """Affiche l'interface d'administration des patterns de reformulation"""
        st.title("Configuration des Reformulations")

        # Afficher les patterns existants
        st.subheader("Patterns de reformulation existants")

        for category, patterns in self.patterns.items():
            with st.expander(f"Catégorie: {category} ({len(patterns)} patterns)"):
                for i, pattern_obj in enumerate(patterns):
                    st.write(f"**Pattern {i + 1}:** `{pattern_obj['pattern']}` → `{pattern_obj['replacement']}`")

        # Ajouter un nouveau pattern
        st.subheader("Ajouter un nouveau pattern")

        # Sélection de la catégorie
        categories = list(self.patterns.keys()) + ["Nouvelle catégorie"]
        selected_category = st.selectbox("Catégorie", categories)

        if selected_category == "Nouvelle catégorie":
            new_category = st.text_input("Nom de la nouvelle catégorie")
            if new_category:
                selected_category = new_category

        # Saisie du pattern et du remplacement
        pattern = st.text_input("Expression régulière (pattern)")
        replacement = st.text_input("Texte de remplacement")

        if st.button("Ajouter le pattern") and pattern and replacement:
            self.add_pattern(selected_category, pattern, replacement)
            st.success(f"Pattern ajouté à la catégorie '{selected_category}'")
            st.rerun()


# Fonction pour intégrer le reformulateur dans l'application principale
def integrate_query_reformulator(help_desk):
    """Intègre le reformulateur de requêtes au help_desk"""
    reformulator = QueryReformulator()

    # Fonction wrapper pour ask_question qui reformule les requêtes
    original_ask = help_desk.ask_question

    def ask_with_reformulation(question, verbose=False, auto_reformulate=True):
        if auto_reformulate:
            # Reformuler automatiquement la question
            reformulated_question = reformulator.reformulate_query(question)
            if verbose and reformulated_question != question:
                print(f"Question reformulée: {reformulated_question}")

            # Utiliser la question reformulée
            answer, sources = original_ask(reformulated_question, verbose)
        else:
            # Utiliser la question originale
            answer, sources = original_ask(question, verbose)

        return answer, sources

    # Remplacer la méthode originale
    help_desk.ask_question_with_reformulation = ask_with_reformulation

    # Ajouter le reformulateur comme attribut
    help_desk.query_reformulator = reformulator

    return help_desk
