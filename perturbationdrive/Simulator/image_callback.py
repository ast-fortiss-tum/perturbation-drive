import pygame


class ImageCallBack:
    def __init__(self, channels: int = 3, rows: int = 240, cols: int = 320):
        """
        Used to display a image on a second screen

        :param channels: number of channels
        :param rows: number of rows
        :param cols: number of cols
        """
        pygame.init()
        ch, row, col = channels, rows, cols

        size = (col * 2, row * 2)
        pygame.display.set_caption("sdsandbox image monitor")
        self.screen: pygame.Surface = pygame.display.set_mode(size, pygame.DOUBLEBUF)
        self.camera_surface = pygame.surface.Surface((col, row), 0, 24).convert()
        self.myfont = pygame.font.SysFont("monospace", 15)

    def screen_print(self, x, y, msg, screen):
        """
        prints a message on the screen
        """
        label = self.myfont.render(msg, 1, (255, 255, 0))
        screen.blit(label, (x, y))

    def display_img(self, img, steering, throttle, perturbation):
        """
        Displays the image and the steering and throttle value
        """
        # swap image axis
        img = img.swapaxes(0, 1)
        # draw frame
        pygame.surfarray.blit_array(self.camera_surface, img)
        camera_surface_2x = pygame.transform.scale2x(self.camera_surface)
        self.screen.blit(camera_surface_2x, (0, 0))
        # steering and throttle value
        self.screen_print(10, 10, "NN(steering): " + steering, self.screen)
        self.screen_print(10, 25, "NN(throttle): " + throttle, self.screen)
        self.screen_print(10, 40, "Perturbation: " + perturbation, self.screen)
        pygame.display.flip()

    def display_waiting_screen(self):
        """
        Displays a waiting screen
        """
        self.screen.fill((0, 0, 0))
        self.screen_print(10, 10, "Waiting for the simulator to start", self.screen)
        pygame.display.flip()

    def display_disconnect_screen(self):
        """
        Displays a disconnect screen
        """
        self.screen.fill((0, 0, 0))
        self.screen_print(10, 10, "Simulator disconnected", self.screen)
        pygame.display.flip()

    def destroy(self):
        """
        Quits the monitor and display
        """
        pygame.display.quit()
        pygame.quit()
