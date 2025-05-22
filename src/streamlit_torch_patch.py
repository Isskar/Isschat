"""Patch to fix Streamlit conflicts with PyTorch.

This module provides a comprehensive fix for the interaction between Streamlit and PyTorch:
1. Completely prevents Streamlit from watching any PyTorch modules
2. Handles the 'no running event loop' error from asyncio
3. Adds proper error handling to avoid crashes when accessing torch.__path__._path

To use this patch, import it before initializing Streamlit.
"""

import sys


class StreamlitTorchPatch:
    """Implements patches for Streamlit/PyTorch compatibility issues."""

    @staticmethod
    def apply_patch():
        """Apply all necessary patches to fix compatibility issues."""
        # Get the streamlit watcher module
        watcher_module = sys.modules.get("streamlit.watcher.local_sources_watcher", None)

        if not watcher_module:
            print("Streamlit watcher module not loaded yet, patch may need to be applied later")
            return False

        try:
            # Patch 1: Modify should_be_added to skip torch modules completely
            orig_should_be_added = getattr(watcher_module, "should_be_added", None)
            if orig_should_be_added:

                def patched_should_be_added(module_name):
                    # Skip any torch-related modules entirely
                    if module_name.startswith("torch") or "torch" in module_name:
                        return False
                    # Use the original function for other modules
                    return orig_should_be_added(module_name)

                # Apply the patch
                setattr(watcher_module, "should_be_added", patched_should_be_added)
                print("✓ Patched should_be_added to ignore torch modules")

            # Patch 2: Override get_module_paths to safely handle torch modules
            orig_get_module_paths = getattr(watcher_module, "get_module_paths", None)
            if orig_get_module_paths:

                def safe_get_module_paths(module):
                    try:
                        # Skip torch modules completely
                        if module.__name__.startswith("torch") or "torch" in module.__name__:
                            return []

                        # Call original with error catching
                        try:
                            return orig_get_module_paths(module)
                        except Exception as e:
                            # Handle any exceptions from the original function
                            if "torch" in str(e):
                                return []
                            # For modules with __file__, just use that
                            if hasattr(module, "__file__") and module.__file__:
                                return [module.__file__]
                            return []
                    except Exception:
                        # Final fallback - don't crash
                        return []

                # Apply the patch
                setattr(watcher_module, "get_module_paths", safe_get_module_paths)
                print("✓ Patched get_module_paths for robust error handling")

            # Patch 3: Override extract_paths to handle errors gracefully
            orig_extract_paths = getattr(watcher_module, "extract_paths", None)
            if orig_extract_paths:

                def safe_extract_paths(module):
                    # Handle torch modules
                    if module.__name__.startswith("torch") or "torch" in module.__name__:
                        if hasattr(module, "__file__") and module.__file__:
                            return [module.__file__]
                        return []

                    # For other modules with __path__, safely extract paths
                    if hasattr(module, "__path__"):
                        try:
                            if hasattr(module.__path__, "_path"):
                                try:
                                    return list(module.__path__._path)
                                except (RuntimeError, AttributeError, TypeError):
                                    pass
                            # Try standard __path__
                            try:
                                return list(module.__path__)
                            except (RuntimeError, AttributeError, TypeError):
                                pass
                        except Exception:
                            pass

                    # Fallback to __file__
                    if hasattr(module, "__file__") and module.__file__:
                        return [module.__file__]

                    # Last resort - empty list
                    return []

                # Apply the patch
                setattr(watcher_module, "extract_paths", safe_extract_paths)
                print("✓ Patched extract_paths for safe path extraction")

            print("✓ StreamlitTorchPatch successfully applied")
            return True

        except Exception as e:
            print(f"Error applying StreamlitTorchPatch: {str(e)}")
            return False


# Apply the patch when this module is imported
success = StreamlitTorchPatch.apply_patch()

# Print status message
if success:
    print("StreamlitTorchPatch: All patches successfully applied")
else:
    print("StreamlitTorchPatch: Patches may need to be applied after streamlit is fully loaded")
