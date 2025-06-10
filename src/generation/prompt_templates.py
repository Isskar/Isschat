"""
Prompt templates for different generation scenarios.
"""

from typing import Dict, Any


class PromptTemplates:
    """Collection of prompt templates for the RAG system."""

    @staticmethod
    def get_default_template() -> str:
        """Get the default prompt template."""
        return """
        You are a professional and friendly virtual assistant named "ISSCHAT".
        Your mission is to help users find information in the Confluence documentation.

        Based on these text excerpts from the documentation:
        -----
        {context}
        -----

        Answer the following question IN FRENCH in a conversational and professional manner.
        Use a friendly but professional tone, as if you were a helpful colleague.
        Be concise but complete. Use French phrases like "je vous suggère de..."
        (I suggest that you...), "vous pourriez..." (you could...), etc.
        If you don't have the information, clearly state so and suggest alternatives.
        IMPORTANT: Always respond in French regardless of the language of the question.

        Question: {question}
        Answer:
        """

    @staticmethod
    def get_concise_template() -> str:
        """Get a template for concise answers."""
        return """
        You are ISSCHAT, a helpful assistant for Confluence documentation.

        Context:
        {context}

        Provide a concise answer in French to: {question}

        Keep your response brief and direct.
        """

    @staticmethod
    def get_detailed_template() -> str:
        """Get a template for detailed answers."""
        return """
        You are ISSCHAT, an expert assistant for Confluence documentation.

        Based on the following documentation excerpts:
        -----
        {context}
        -----

        Provide a comprehensive answer in French to the following question: {question}

        Please include:
        - A clear explanation of the topic
        - Step-by-step instructions if applicable
        - Any relevant warnings or considerations
        - Suggestions for further reading if appropriate

        Use a professional but approachable tone in French.
        """

    @staticmethod
    def get_troubleshooting_template() -> str:
        """Get a template for troubleshooting scenarios."""
        return """
        You are ISSCHAT, a technical support assistant for Confluence documentation.

        Documentation context:
        {context}

        The user is experiencing an issue: {question}

        Provide troubleshooting guidance in French including:
        1. Possible causes of the issue
        2. Step-by-step resolution steps
        3. Prevention tips
        4. When to escalate to technical support

        Be systematic and helpful in your response.
        """

    @staticmethod
    def get_template_by_type(template_type: str = "default") -> str:
        """
        Get a prompt template by type.

        Args:
            template_type: Type of template to retrieve

        Returns:
            Prompt template string
        """
        templates = {
            "default": PromptTemplates.get_default_template(),
            "concise": PromptTemplates.get_concise_template(),
            "detailed": PromptTemplates.get_detailed_template(),
            "troubleshooting": PromptTemplates.get_troubleshooting_template(),
        }

        return templates.get(template_type, templates["default"])  # type : ignore

    @staticmethod
    def get_available_templates() -> Dict[str, str]:
        """
        Get list of available template types.

        Returns:
            Dictionary mapping template names to descriptions
        """
        return {
            "default": "Standard conversational template",
            "concise": "Brief and direct responses",
            "detailed": "Comprehensive explanations",
            "troubleshooting": "Technical support scenarios",
        }

    @staticmethod
    def customize_template(base_template: str, customizations: Dict[str, Any]) -> str:
        """
        Customize a template with specific parameters.

        Args:
            base_template: Base template string
            customizations: Dictionary of customization parameters

        Returns:
            Customized template string
        """
        template = base_template

        # Apply customizations
        if "assistant_name" in customizations:
            template = template.replace("ISSCHAT", customizations["assistant_name"])

        if "language" in customizations and customizations["language"] != "french":
            # Replace French-specific instructions
            if customizations["language"] == "english":
                template = template.replace("IN FRENCH", "IN ENGLISH")
                template = template.replace("in French", "in English")
                template = template.replace("en français", "in English")

        if "tone" in customizations:
            tone = customizations["tone"]
            if tone == "formal":
                template = template.replace("friendly", "formal")
                template = template.replace("conversational", "professional")
            elif tone == "casual":
                template = template.replace("professional", "casual")

        return template
