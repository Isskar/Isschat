from src.help_desk import HelpDesk


class TestHelpDesk:
    @classmethod
    def setup_class(cls):
        cls.prompt = HelpDesk.get_template()

    def test_prompt_is_in_french(self):
        assert "IMPORTANT: Always respond in French regardless of the language of the question" in self.prompt

    def test_prompt_contains_variables(self):
        assert "{context}" in self.prompt
        assert "{question}" in self.prompt
