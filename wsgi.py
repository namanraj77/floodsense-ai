"""
Vercel entrypoint — loads the Flask app from backend/app.py
Uses importlib to avoid naming conflict with this file.
"""
import sys
import os
import importlib.util

# Add backend/ to path (so train_models.py is importable)
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_dir)

# Load backend/app.py as 'flask_app' to avoid clashing with this module name
spec = importlib.util.spec_from_file_location(
    'flask_app',
    os.path.join(backend_dir, 'app.py')
)
module = importlib.util.module_from_spec(spec)
sys.modules['flask_app'] = module
spec.loader.exec_module(module)

# Vercel looks for a variable named 'app'
app = module.app
