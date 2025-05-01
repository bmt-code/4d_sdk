import setuptools

setuptools.setup(
    name="stereo_4d",
    version="0.0.1",
    python_requires=">=3.8",
    packages=["stereo_4d"],
    install_requires=[
        "opencv-python",
        "numpy",
        "zmq",
    ],
)
