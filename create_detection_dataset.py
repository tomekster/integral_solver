import glob
import numpy as np
import cv2
import random 

HEIGHT = 400
WIDTH = 400
EXAMPLE_SIZE = 32
STRIDE = 3
TOTAL = EXAMPLE_SIZE + 2 * STRIDE

hasy_examples = glob.glob('../HASYv2/hasy-data/*.png')
random.shuffle(hasy_examples)

while hasy_examples:
    canva = np.full((HEIGHT, WIDTH), 255, np.uint8)
    for i in range(HEIGHT // TOTAL):
        for j in range(WIDTH // TOTAL):
            if not hasy_examples:
                continue
            
            offsets = [-1, 0, 1]
            dx = random.choice(offsets)
            dy = random.choice(offsets)
            ypos = i*TOTAL + STRIDE + dy
            xpos = j*TOTAL + STRIDE + dx
            
            example_path = hasy_examples.pop()
            example = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
            
            canva[  ypos : ypos + EXAMPLE_SIZE, 
                    xpos : xpos + EXAMPLE_SIZE] = example

    cv2.imshow('blank', canva)
    cv2.waitKey(0)

