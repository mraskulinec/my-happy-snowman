from utils import *
from ray import *
from cli import render
from PIL import Image

"""
STILL TODO: 

1. make the tree look better / add a cone primitive
2. add more details to the snowman
3. make a crescent moon in the nightsky
4. add some cube-shaped "presents" around the tree
5. add christmas ornaments to the tree

"""
load_img_path = './ornament.png'
img = Image.open(load_img_path)
im = np.array(img)

a = np.log(np.array([0.4, 0.4, 0.2])**(-1))
crystal = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90,
                   k_m=0.3, di=True, n=1.5, a=a)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)
red = Material(vec([0.8, 0.2, 0.2]), k_m=0.5)
green = Material(vec([0.02, 0.5, 0.02]), k_m=0.5)
tree = Material(vec([0.01, 0.1, 0.01]), k_m=0.5)
brown = Material(vec([0.1, 0.1, 0.1]), k_m=0.5)
gold = Material(vec([.5, .5, 0.1]), k_m=0.5)
snow = Material(vec([1, 1, 1.5]), k_m=0.5)
sparkle_red = Material(vec([0.2, 0.2, 0.2]), k_m=0.5, img=im)

scene = Scene([
    # Christmas balls
    Sphere(vec([-.9, -0.4, -0.4]), 0.1, sparkle_red),
    Sphere(vec([-.8, -0.1, -.4]), 0.1, crystal),
    Sphere(vec([-0.6, 0.1, -.7]), 0.1, sparkle_red),
    Sphere(vec([-0.8, 0.3, -.3]), 0.1, crystal),
    Sphere(vec([-0.7, 0.4, -.7]), 0.1, sparkle_red),
    Sphere(vec([-0.5, 0.6, -.7]), 0.1, crystal),
    Sphere(vec([-0.3, -0.5, -.5]), 0.1, sparkle_red),
    Sphere(vec([-0.9, 0.8, -1]), 0.1, sparkle_red),
    # Sphere(vec([0.1, 0.7, 0]), 0.1, crystal),

    # Snowman :)
    Sphere(vec([1, -0.8, 0]), 0.2, snow),
    Sphere(vec([1, -0.5, 0]), 0.15, snow),
    Sphere(vec([1, -0.3, 0]), 0.1, snow),
    Sphere(vec([1, -0.22, 0]), 0.05, red),
    Sphere(vec([1, -0.275, 0.1]), 0.01, brown),  # my attempt to make eyes
    Sphere(vec([1.05, -0.275, 0.05]), 0.01, brown),

    # Moon
    Sphere(vec([-10, 0, -30]), 1, gold),  # not visible rn :(

    # Tree Code
    Sphere(vec([-1, -0.7, -1]), 0.3, brown),
    Sphere(vec([-1, -0.6, -1]), 0.55, tree),
    Sphere(vec([-1, -0.5, -1]), 0.53, tree),
    Sphere(vec([-1, -0.45, -1]), 0.52, tree),
    Sphere(vec([-1, -0.4, -1]), 0.51, gold),
    Sphere(vec([-1, -0.3, -1]), 0.49, tree),
    Sphere(vec([-1, -0.1, -1]), 0.47, tree),
    Sphere(vec([-1, 0, -1]), 0.45, tree),
    Sphere(vec([-1, 0.1, -1]), 0.44, tree),
    Sphere(vec([-1, -0.25, -1]), 0.435, tree),
    Sphere(vec([-1, 0.2, -1]), 0.43, gold),
    Sphere(vec([-1, 0.3, -1]), 0.42, tree),
    Sphere(vec([-1, 0.4, -1]), 0.41, tree),
    Sphere(vec([-1, 0.5, -1]), 0.37, gold),
    Sphere(vec([-1, 0.6, -1]), 0.35, tree),
    Sphere(vec([-1, 0.7, -1]), 0.32, tree),
    Sphere(vec([-1, 0.8, -1]), 0.29, tree),

    # Christmas Tree Topper
    Sphere(vec([-1, 1.1, -1]), 0.1, gold),

    # Ground floor
    Sphere(vec([0, -201, 0]), 200, snow),
],
    bg_color=vec([0.001, 0.01, 0.1])
)

lights = [
    PointLight(vec([12, 10, 5]), vec([100, 100, 100])),
    PointLight(vec([-40, 10, -5]), vec([200, 200, 200])),
    PointLight(vec([-10, 0, -30]), vec([1000000000, 1000000000, 1000000000])),
    AmbientLight(0.05),
]

camera = Camera(vec([3, 1.2, 5]), target=vec(
    [0, 0, 0]), vfov=24, aspect=16/9)

render(camera, scene, lights)
