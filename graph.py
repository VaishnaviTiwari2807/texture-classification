import tensorflow as tf
import webbrowser
import subprocess
# Define the graph
# a = tf.constant(2, name='a')
# b = tf.constant(3, name='b')
# c = tf.add(a, b, name='sum')

# Write the graph data to disk
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# writer.close()

# Launch TensorBoard
webbrowser.open('http://localhost:6006/?darkMode=true#timeseries')
subprocess.Popen (['tensorboard', '--logdir','/logs'])