import re
import json
import os


class QueryReformulator:
    """Automatically reformulates queries to improve response quality"""

    def __init__(self, cache_path="./cache"):
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
        self.reformulation_patterns_file = os.path.join(
            cache_path, "reformulation_patterns.json"
        )
        self.patterns = self._load_patterns()

    def _load_patterns(self):
        """Load reformulation patterns from cache or create default patterns"""
        if os.path.exists(self.reformulation_patterns_file):
            try:
                with open(self.reformulation_patterns_file, "r") as f:
                    return json.load(f)
            except:
                return self._create_default_patterns()
        else:
            return self._create_default_patterns()

    def _create_default_patterns(self):
        """Create default reformulation patterns"""
        default_patterns = {
            "expansion": [
                {"pattern": r"\b(\w+)\b", "replacement": r"detailed context \1"},
                {"pattern": r"how (.+)", "replacement": r"explain in detail how \1"},
                {
                    "pattern": r"what is (.+)",
                    "replacement": r"define and explain in detail \1",
                },
                {"pattern": r"why (.+)", "replacement": r"explain the reasons why \1"},
            ],
            "precision": [
                {
                    "pattern": r"information about (.+)",
                    "replacement": r"specific and detailed information about \1",
                },
                {
                    "pattern": r"talk about (.+)",
                    "replacement": r"explain in detail \1 with concrete examples",
                },
                {
                    "pattern": r"I want to know (.+)",
                    "replacement": r"provide detailed information about \1",
                },
            ],
            "context": [
                {
                    "pattern": r"(.+) project (.+)",
                    "replacement": r"\1 project \2 in the context of our company",
                },
                {
                    "pattern": r"(.+) process (.+)",
                    "replacement": r"\1 process \2 according to our internal documentation",
                },
            ],
        }

        with open(self.reformulation_patterns_file, "w") as f:
            json.dump(default_patterns, f)

        return default_patterns

    def _save_patterns(self):
        """Save the reformulation patterns to cache"""
        with open(self.reformulation_patterns_file, "w") as f:
            json.dump(self.patterns, f)

    def add_pattern(self, category, pattern, replacement):
        """Add a new reformulation pattern"""
        if category not in self.patterns:
            self.patterns[category] = []

        self.patterns[category].append({"pattern": pattern, "replacement": replacement})

        self._save_patterns()

    def reformulate_query(self, query, strategy="auto"):
        """Reformulate a query according to the specified strategy"""
        original_query = query

        if strategy == "auto":
            # Automatically determine the best strategy
            if len(query.split()) <= 3:
                strategy = "expansion"  # Short query -> expansion
            elif any(word in query.lower() for word in ["how", "why", "what", "who"]):
                strategy = "precision"  # Question -> precision
            else:
                strategy = "context"  # Default -> context

        # Apply patterns from the chosen strategy
        if strategy in self.patterns:
            for pattern_obj in self.patterns[strategy]:
                pattern = pattern_obj["pattern"]
                replacement = pattern_obj["replacement"]
                query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

        # Avoid identical or too long reformulations
        if query == original_query or len(query) > 3 * len(original_query):
            return original_query

        return query

    def suggest_reformulations(self, query, max_suggestions=3):
        """Suggests multiple possible reformulations"""
        suggestions = []

        for strategy in self.patterns:
            reformulation = self.reformulate_query(query, strategy)
            if reformulation != query and reformulation not in suggestions:
                suggestions.append(reformulation)

        return suggestions[:max_suggestions]

    def render_reformulation_widget(self, st, query, callback=None):
        """Display a reformulation widget with suggestions"""
        suggestions = self.suggest_reformulations(query)

        if not suggestions:
            return query

        st.write("---")
        st.write("### Suggested Reformulations")
        st.write("Your query could be more precise. Try these reformulations:")

        selected_query = query

        for i, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"reform_{i}"):
                selected_query = suggestion
                if callback:
                    callback(suggestion)

        return selected_query

    def render_admin_interface(self, st):
        """Display the administration interface for reformulation patterns"""
        st.title("Reformulation Configuration")

        st.subheader("Existing reformulation patterns")

        for category, patterns in self.patterns.items():
            with st.expander(f"Category: {category} ({len(patterns)} patterns)"):
                for i, pattern_obj in enumerate(patterns):
                    st.write(
                        f"**Pattern {i + 1}:** `{pattern_obj['pattern']}` → `{pattern_obj['replacement']}`"
                    )

        st.subheader("Add a new pattern")

        categories = list(self.patterns.keys()) + ["New category"]
        selected_category = st.selectbox("Category", categories)

        if selected_category == "New category":
            new_category = st.text_input("Name of the new category")
            if new_category:
                selected_category = new_category

        # Pattern and replacement input
        pattern = st.text_input("Regular expression (pattern)")
        replacement = st.text_input("Replacement text")

        if st.button("Add pattern") and pattern and replacement:
            self.add_pattern(selected_category, pattern, replacement)
            st.success(f"Pattern added to category '{selected_category}'")
            st.rerun()


# Function to integrate the reformulator into the main application
def integrate_query_reformulator(help_desk):
    """Integrates the query reformulator into the help_desk"""
    reformulator = QueryReformulator()

    # Wrapper function for ask_question that reformulates queries
    original_ask = help_desk.ask_question

    def ask_with_reformulation(question, verbose=False, auto_reformulate=True):
        if auto_reformulate:
            # Automatically reformulate the question
            reformulated_question = reformulator.reformulate_query(question)
            if verbose and reformulated_question != question:
                print(f"Reformulated question: {reformulated_question}")

            # Use the reformulated question
            answer, sources = original_ask(reformulated_question, verbose)
        else:
            # Use the original question
            answer, sources = original_ask(question, verbose)

        return answer, sources

    # Replace the original method
    help_desk.ask_question_with_reformulation = ask_with_reformulation

    # Add the reformulator as an attribute
    help_desk.query_reformulator = reformulator

    return help_desk
