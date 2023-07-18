# PRESS ENTER TO SUBMIT TO NEURAL NETWORK
# PRESS BACKSPACE TO DISCARD

import matplotlib.pyplot as plt
import pygame
import cv2 as cv
import numpy as np
import tensorflow as tf

# Loading pre-trained model
model = tf.keras.models.load_model("digit_classification_model.h5")

pygame.init()
pygame.font.init()
FPS = 300
delta = 100
WIDTH, HEIGHT = 28 * 20, 28 * 20
WIN = pygame.display.set_mode((WIDTH, HEIGHT + delta), pygame.HWSURFACE)
pygame.display.set_caption("Digit Recogniser")

# Initializing colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
ORANGE = (255, 216, 168)
YELLOW = (246, 202, 72)


def prediction_msg(msg):
    pygame.draw.rect(WIN, WHITE, pygame.Rect(0, HEIGHT, WIDTH, delta - 25))
    font = pygame.font.SysFont("Cambria", 40)
    text = font.render(f"I predict it as: {msg}", True, BLACK)
    WIN.blit(text, (160, HEIGHT + 50))


def call_NN():
    # get colour values of all the pixels
    pixel_arr = []
    for val_y in range(HEIGHT):
        for val_x in range(WIDTH):
            value = WIN.get_at((val_x, val_y))
            if value != (255, 255, 255, 255):
                pixel_arr.append((255 - value[0]) / 255)
            else:
                pixel_arr.append(0)

    pixel_arr = np.array(pixel_arr).reshape((WIDTH, HEIGHT))
    img = cv.resize(pixel_arr, dsize=(28, 28), interpolation=cv.INTER_CUBIC)
    img = np.expand_dims(img, axis=0)
    pred_arr = model.predict(img)
    number_detected = np.argmax(pred_arr)
    print([round(val, 2) for val in pred_arr[0]], number_detected)
    prediction_msg(number_detected)

    # Uncomment these lines to visualize the modified image
    #fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100))

    #print(img.shape, number_detected)
    #ax.imshow(img[0])
    #plt.show()


def draw_digit(x_pos, y_pos):
    width = 25
    # Rectangular tip
    #pygame.draw.rect(WIN, BLACK, pygame.Rect(x_pos, y_pos, width, width))
    # Circular tip
    pygame.draw.circle(WIN, BLACK, (x_pos, y_pos), width)


def main():
    run = True
    is_pressed, submit, clear = False, False, True
    clock = pygame.time.Clock()

    # Initializing window
    WIN.fill(WHITE)

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            else:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not clear:
                        WIN.fill(WHITE)
                        clear = True
                    is_pressed = True
                if event.type == pygame.MOUSEBUTTONUP:
                    is_pressed = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        if not clear:
                            WIN.fill(WHITE)
                            clear = True
                        else:
                            submit = True
                    if event.key == pygame.K_BACKSPACE:
                        WIN.fill(WHITE)

        if is_pressed:
            x_pos, y_pos = pygame.mouse.get_pos()
            if y_pos <= HEIGHT:
                draw_digit(x_pos, y_pos)

        if submit:
            call_NN()
            submit = False
            clear = False

        pygame.display.update()


if __name__ == "__main__":
    main()
