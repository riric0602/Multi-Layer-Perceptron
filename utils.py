from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent

def close_on_key(event: KeyEvent) -> None:
    """
    Close the window when the ESC key is pressed
    :param event: keyboard event
    :return:
    """
    if event.key == 'escape':
        plt.close(event.canvas.figure)