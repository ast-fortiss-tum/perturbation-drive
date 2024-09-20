import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class ImageCallBack:
    def __init__(self, channels: int = 3, rows: int = 240, cols: int = 320):
        """
        Used to display an image on a second screen

        :param channels: number of channels
        :param rows: number of rows
        :param cols: number of cols
        """
        self.channels = channels
        self.rows = rows
        self.cols = cols
        self.window = tk.Tk()
        self.window.title("sdsandbox image monitor")
        self.canvas = tk.Canvas(self.window, width=cols * 2, height=rows * 2)
        self.canvas.pack()
        self.label_text = tk.StringVar()
        self.label = tk.Label(
            self.window,
            textvariable=self.label_text,
            font=("Monospace", 15),
            fg="yellow",
            bg="black",
        )
        self.label.pack()

    def screen_print(self, msg):
        """
        prints a message on the screen
        """
        self.label_text.set(msg)

    def display_img(self, img, steering, throttle, perturbation):
        """
        Displays the image and the steering and throttle value
        """
        # Ensure the image is in uint8 format
        img = img.astype(np.uint8)
        # swap image axis
        img = Image.fromarray(img)
        img = img.resize((self.cols * 2, self.rows * 2))
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.screen_print(
            f"NN(steering): {steering}\nNN(throttle): {throttle}\nPerturbation: {perturbation}"
        )
        self.window.update()

    def display_waiting_screen(self):
        """
        Displays a waiting screen
        """
        self.canvas.create_rectangle(0, 0, self.cols * 2, self.rows * 2, fill="red")
        self.screen_print("Waiting for the simulator to start")
        self.window.update()

    def display_disconnect_screen(self):
        """
        Displays a disconnect screen
        """
        self.canvas.create_rectangle(0, 0, self.cols * 2, self.rows * 2, fill="black")
        self.screen_print("Simulator disconnected")
        self.window.update()

    def destroy(self):
        """
        Quits the monitor and display
        """
        self.window.destroy()
