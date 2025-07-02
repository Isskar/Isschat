from unittest.mock import Mock
from src.storage.storage_interface import StorageInterface
from src.storage.local_storage import LocalStorage


class TestStorageInterface:
    def test_load_text_file_success(self):
        storage = Mock(spec=StorageInterface)
        storage.read_file.return_value = b"test content"

        result = StorageInterface.load_text_file(storage, "test.txt")

        assert result == "test content"
        storage.read_file.assert_called_once_with("test.txt")

    def test_load_text_file_failure(self):
        """Test text file loading failure."""
        storage = Mock(spec=StorageInterface)
        storage.read_file.side_effect = Exception("File not found")

        result = StorageInterface.load_text_file(storage, "nonexistent.txt")

        assert result is None

    def test_append_jsonl_data_new_file(self):
        storage = Mock(spec=StorageInterface)
        storage.file_exists.return_value = False
        storage.write_file.return_value = True

        data = {"key": "value", "number": 42}
        result = StorageInterface.append_jsonl_data(storage, "test.jsonl", data)

        assert result is True
        storage.write_file.assert_called_once()

    def test_append_jsonl_data_existing_file(self):
        """Test appending JSON data to existing file."""
        storage = Mock(spec=StorageInterface)
        storage.file_exists.return_value = True
        existing_content = '{"existing": "data"}\n'
        storage.read_file.return_value = existing_content.encode("utf-8")
        storage.write_file.return_value = True

        data = {"new": "data"}
        result = StorageInterface.append_jsonl_data(storage, "test.jsonl", data)

        assert result is True


class TestLocalStorage:
    """Tests for local storage."""

    def test_init_creates_base_directory(self, temp_dir):
        """Test initialization creates base directory."""
        base_path = temp_dir / "storage"
        storage = LocalStorage(str(base_path))

        assert storage.base_path == base_path
        assert base_path.exists()

    def test_write_and_read_file(self, temp_dir):
        """Test writing and reading a file."""
        storage = LocalStorage(str(temp_dir))
        test_data = b"Hello, World!"

        result = storage.write_file("test.txt", test_data)
        assert result is True

        read_data = storage.read_file("test.txt")
        assert read_data == test_data

    def test_file_exists(self, temp_dir):
        """Test file existence check."""
        storage = LocalStorage(str(temp_dir))

        assert not storage.file_exists("nonexistent.txt")

        storage.write_file("existing.txt", b"content")
        assert storage.file_exists("existing.txt")

    def test_delete_file(self, temp_dir):
        """Test file deletion."""
        storage = LocalStorage(str(temp_dir))

        storage.write_file("to_delete.txt", b"content")
        assert storage.file_exists("to_delete.txt")

        result = storage.delete_file("to_delete.txt")
        assert result is True
        assert not storage.file_exists("to_delete.txt")

    def test_create_directory(self, temp_dir):
        """Test directory creation."""
        storage = LocalStorage(str(temp_dir))

        result = storage.create_directory("new_dir")
        assert result is True
        assert storage.directory_exists("new_dir")

    def test_list_files(self, temp_dir):
        """Test file listing."""
        storage = LocalStorage(str(temp_dir))

        storage.write_file("file1.txt", b"content1")
        storage.write_file("file2.txt", b"content2")
        storage.write_file("data.json", b'{"key": "value"}')

        all_files = storage.list_files(".")
        assert len(all_files) == 3
        assert "file1.txt" in all_files
        assert "file2.txt" in all_files
        assert "data.json" in all_files
