import tensorflow as tf
import webbrowser
import subprocess


# Launch TensorBoard
webbrowser.open('http://localhost:6006/?darkMode=true#timeseries')
subprocess.Popen (['tensorboard', '--logdir','/logs'])