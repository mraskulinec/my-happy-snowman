from utils import *
from ray import *
from cli import render
from PIL import Image
import math

load_img_path = './cube.png'
img = Image.open(load_img_path)
im = np.array(img)


# x = np.log(10)
# a = np.array([x, x, x])
a = np.log(np.array([0.4, 0.4, 0.2])**(-1))
tan = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90,
               k_m=0.3, di=True, n=1.5, a=a)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

# m = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
# m = np.identity(3)
scene = Scene([
    Sphere(vec([-0.7, 0, 0]), 0.5, tan),
    Sphere(vec([0.7, 0, 0]), 0.5, blue),
    Sphere(vec([0, -40, 0]), 39.5, gray),
    Cone(vec([3, 3, 3]), vec([0, 0, -1]), math.pi/16, 3, tan),
])

lights = [
    PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3, 1.2, 5]), target=vec(
    [0, -0.4, 0]), vfov=24, aspect=16/9)

render(camera, scene, lights)
