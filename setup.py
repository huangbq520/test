"""
中文深度学习代码助手安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dl-assistant",
    version="0.1.0",
    author="深度学习助手团队",
    author_email="",
    description="中文深度学习代码助手 - 简化深度学习开发流程",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/huangbq520/test",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "tqdm>=4.60.0",
    ],
)
