import matplotlib.pyplot as plt
import numpy as np


class SpiralVisualizer:
    def __init__(self, X, y, title="Training Data"):
        self.title = title

        # Turn on interactive mode
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.manager.set_window_title(title)

        # Set up grid boundaries
        # 0.05 step size during training for speed
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        self.xx, self.yy = np.meshgrid(
            np.arange(x_min, x_max, 0.05),
            np.arange(y_min, y_max, 0.05)
        )

        self.grid_inputs = np.c_[self.xx.ravel(), self.yy.ravel()]

        self.X = X
        self.y = y


    def update(self, epoch, predict_func):
        """Call this inside the training loop to update the plot live."""

        self.ax.clear()

        # Get predictions for the grid using the model
        grid_preds = predict_func(self.grid_inputs)
        grid_preds = grid_preds.reshape(self.xx.shape)

        # Draw contours and data points
        self.ax.contourf(
            self.xx,
            self.yy,
            grid_preds,
            alpha=0.4,
            cmap=plt.cm.Spectral
        )

        self.ax.scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=self.y,
            s=40,
            cmap=plt.cm.Spectral,
            edgecolors='k'
        )

        self.ax.set_title(f"{self.title} — Epoch {epoch}")
        self.ax.set_xlabel("X coordinate")
        self.ax.set_ylabel("Y coordinate")

        # Redraw and refresh
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.001)


    def show_final(self, predict_func):
        """
        Create a clean, high-resolution final plot.

        Does NOT call plt.show() so multiple final plots can be
        created before displaying them together.
        """

        # Close the live training window
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

        # High-resolution grid
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.01),
            np.arange(y_min, y_max, 0.01)
        )

        grid_inputs = np.c_[xx.ravel(), yy.ravel()]
        grid_preds = predict_func(grid_inputs).reshape(xx.shape)

        # Create final figure
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.canvas.manager.set_window_title(self.title)

        ax.contourf(
            xx,
            yy,
            grid_preds,
            alpha=0.4,
            cmap=plt.cm.Spectral
        )

        ax.scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=self.y,
            s=40,
            cmap=plt.cm.Spectral,
            edgecolors='k'
        )

        ax.set_title(f"{self.title} — Final Decision Boundaries")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")

        return fig