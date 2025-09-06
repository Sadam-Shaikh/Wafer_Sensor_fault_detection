# Import configuration from root config

try:
    # Try importing from root config first
    import sys
    import os
    
    # Add root directory to path if needed
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    
    # Import all variables from root config
    from config.config import *
except ImportError as e:
    # Log the error
    print(f"Error importing root config: {e}")
    raise