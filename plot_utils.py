import matplotlib
matplotlib.use('TkAgg')
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch


class HandlerRect(HandlerPatch):
    """Class implementing colored rectangle in the legend"""
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        x = width//2
        y = 0
        w = 10
        h = 3.5

        # create
        p = patches.Rectangle(xy=(x, 3), width=w, height=h)

        # update with data from original object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]


class HandlerCircle(HandlerPatch):
    """Class implementing colored circle in the legend"""
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        x = width//2
        y = 0
        w = h = 10

        # create
        p = patches.Circle(xy=(x + 5, 5.25), radius=4.)

        # update with data from original object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]