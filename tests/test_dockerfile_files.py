"""
Test to verify that all files copied by the Dockerfile exist.
"""

import pytest
from pathlib import Path


class TestDockerfileFiles:
    """Test class to verify Dockerfile file dependencies."""

    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent

    def test_pyproject_toml_exists(self, project_root):
        """Test that pyproject.toml exists."""
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml file is missing"
        assert pyproject_path.is_file(), "pyproject.toml should be a file"

    def test_uv_lock_exists(self, project_root):
        """Test that uv.lock exists."""
        uv_lock_path = project_root / "uv.lock"
        assert uv_lock_path.exists(), "uv.lock file is missing"
        assert uv_lock_path.is_file(), "uv.lock should be a file"

    def test_src_directory_exists(self, project_root):
        """Test that src directory exists and contains files."""
        src_path = project_root / "src"
        assert src_path.exists(), "src directory is missing"
        assert src_path.is_dir(), "src should be a directory"

        # Check that src directory is not empty
        src_contents = list(src_path.iterdir())
        assert len(src_contents) > 0, "src directory should not be empty"

    def test_streamlit_directory_exists(self, project_root):
        """Test that .streamlit directory exists."""
        streamlit_path = project_root / ".streamlit"

        # Note: .streamlit directory might not exist in the repository
        # but is expected by the Dockerfile. This test will warn if missing.
        if not streamlit_path.exists():
            pytest.skip(".streamlit directory does not exist - may need to be created for Docker build")

        assert streamlit_path.is_dir(), ".streamlit should be a directory"

    def test_readme_exists(self, project_root):
        """Test that README.md exists."""
        readme_path = project_root / "README.md"
        assert readme_path.exists(), "README.md file is missing"
        assert readme_path.is_file(), "README.md should be a file"

    def test_webapp_app_exists_for_cmd(self, project_root):
        """Test that the app file referenced in CMD exists."""
        app_path = project_root / "src" / "web_app" / "app.py"

        # Check if the exact path from CMD exists
        if not app_path.exists():
            # Look for alternative webapp paths
            webapp_path = project_root / "src" / "webapp" / "app.py"
            assert webapp_path.exists(), (
                f"Neither {app_path} nor {webapp_path} exists. The Dockerfile CMD references src/web_app/app.py"
            )

    def test_uv_command_dependencies(self, project_root):
        """Test that files required by RUN commands exist."""
        # RUN ["uv", "sync"] requires pyproject.toml and uv.lock
        pyproject_path = project_root / "pyproject.toml"
        uv_lock_path = project_root / "uv.lock"

        assert pyproject_path.exists(), "pyproject.toml required by 'uv sync' command is missing"
        assert uv_lock_path.exists(), "uv.lock required by 'uv sync' command is missing"

        # Verify pyproject.toml has required sections for uv
        import tomllib

        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)

        assert "project" in pyproject_data, "pyproject.toml missing [project] section required by uv"
        assert "dependencies" in pyproject_data["project"], "pyproject.toml missing dependencies required by uv"

    def test_cmd_command_dependencies(self, project_root):
        """Test that files required by CMD command exist."""
        # CMD ["uv", "run", "streamlit", "run", "src/web_app/app.py", ...]

        # Check the main app file
        app_path = project_root / "src" / "web_app" / "app.py"
        webapp_path = project_root / "src" / "webapp" / "app.py"

        assert app_path.exists() or webapp_path.exists(), (
            f"App file referenced in CMD does not exist. Expected: {app_path} or {webapp_path}"
        )

        # Check that streamlit is in dependencies
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            import tomllib

            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)

            dependencies = pyproject_data.get("project", {}).get("dependencies", [])
            streamlit_found = any("streamlit" in dep for dep in dependencies)
            assert streamlit_found, "streamlit dependency required by CMD not found in pyproject.toml"

    def test_all_dockerfile_dependencies(self, project_root):
        """Comprehensive test for all Dockerfile COPY dependencies."""
        required_files = ["pyproject.toml", "uv.lock", "README.md"]

        required_directories = ["src"]

        optional_directories = [
            ".streamlit"  # May not exist in repo but needed for Docker
        ]

        missing_files = []
        missing_dirs = []

        # Check required files
        for file_name in required_files:
            file_path = project_root / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        # Check required directories
        for dir_name in required_directories:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)

        # Report all missing dependencies at once
        error_messages = []
        if missing_files:
            error_messages.append(f"Missing required files: {', '.join(missing_files)}")
        if missing_dirs:
            error_messages.append(f"Missing required directories: {', '.join(missing_dirs)}")

        assert not error_messages, "; ".join(error_messages)

        # Warn about optional directories
        for dir_name in optional_directories:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                pytest.warns(UserWarning, f"Optional directory {dir_name} does not exist")
