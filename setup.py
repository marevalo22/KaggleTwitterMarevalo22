import setuptools
import os
import re

# Function to read the contents of your README file
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    return long_description

# Function to read requirements from requirements.txt
def get_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = f.read().splitlines()
    return requirements

# Function to get version from twitter_emotions_cli/__init__.py
def get_version():
    init_py_path = os.path.join(os.path.dirname(__file__), "twitter_emotions_cli", "__init__.py")
    try:
        with open(init_py_path, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", content, re.M)
            if match:
                return match.group(1)
    except FileNotFoundError:
        pass # __init__.py might not exist yet or during initial setup phases
    
    # Fallback version if __init__.py parsing fails or during early build stages
    print("Warning: Could not parse version from __init__.py, using default 0.1.0")
    return "0.1.0"


setuptools.setup(
    name="twitter-emotions-cli",
    version=get_version(), # Gets version from __init__.py
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="A CLI tool for classifying emotions in tweets.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/twitter_emotions_package",  # Replace with your GitHub repo URL later
    license="MIT", # Or any other license you prefer
    packages=setuptools.find_packages(exclude=["tests*", "*.tests", "*.tests.*", "tests"]),
    # This tells setuptools to find your 'twitter_emotions_cli' package automatically
    
    install_requires=get_requirements(), # Reads dependencies from requirements.txt
    
    # This part is crucial to include your model files with the package
    include_package_data=True,
    package_data={
        # If 'trained_model_files' is under 'twitter_emotions_cli', this tells setuptools to include all files in it
        "twitter_emotions_cli": ["trained_model_files/*", "trained_model_files/.*"],
    },
    
    # This creates the command-line script
    entry_points={
        "console_scripts": [
            "twitter-emotions-predict=twitter_emotions_cli.main:run",
            # This means: create a command 'twitter-emotions-predict'
            # that will execute the 'run' function in 'twitter_emotions_cli/main.py'
        ],
    },
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires='>=3.8', # Specify your Python version requirement
)
