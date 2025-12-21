from setuptools import setup, find_packages

setup(
    name="cosmetic_tutor",
    version="0.1.0",
    description="A cosmetic makeup analysis application using AI",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python",
        "mediapipe==0.10.5",
        "numpy",
    ],
)
