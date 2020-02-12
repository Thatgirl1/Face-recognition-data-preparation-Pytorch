import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

if __name__ == '__main__':
    img1 = Image.open('./2.jpg').convert("L")
    mean = np.mean(img1)
    gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
    print(gamma_val)
    img1 = np.array(img1)
    print(img1)
    fI =img1 / 255.0
    gamma = 0.8
    Oc = np.power(fI, gamma_val)
    data = Oc * 255.0
    print(data.dtype)
    img = Image.fromarray(data.astype('uint8')).convert("L")
    print(img)
    img.save("./test.jpg")

