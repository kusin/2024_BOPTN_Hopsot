# lib data visualizations
import matplotlib.pyplot as plt
# --------------------------------------------------------------

def lineplot1(x, y, label):
  
  # membuat frame
  fig, ax = plt.subplots(figsize = (10,5))
  
  # # membuat time series plot
  ax.plot(x, y, color="tab:blue", label=label, linewidth=2.5)
  ax.set_title("", fontsize=14)
  ax.set_xlabel("", fontsize=12)
  ax.set_ylabel("", fontsize=12)
  ax.legend(loc="upper left")
  ax.grid(True)

  # return values
  return plt.show()
# --------------------------------------------------------------

def lineplot2(x1, y1, x2, y2, label1, label2):
  # membuat frame
  fig, ax = plt.subplots(figsize = (10,5))
  
  # # membuat time series plot
  ax.plot(x1, y1, color="tab:blue", label=label1, linewidth=2.5)
  ax.plot(x2, y2, color="tab:red", label=label2, linewidth=2.5)
  ax.set_title("", fontsize=14)
  ax.set_xlabel("", fontsize=12)
  ax.set_ylabel("", fontsize=12)
  ax.legend(loc="upper left")
  ax.grid(True)

  # return values
  return plt.show()