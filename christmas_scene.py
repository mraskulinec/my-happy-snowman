from utils import *
from ray import *
from cli import render

# x = np.log(10)
# a = np.array([x, x, x])
a = np.log(np.array([0.4, 0.4, 0.2])**(-1))
tan = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90,
               k_m=0.3, di=True, n=1.5, a=a)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)
red = Material(vec([0.8, 0.2, 0.2]), k_m=0.5)
green = Material(vec([0.02, 0.5, 0.02]), k_m=0.5)
tree = Material(vec([0.01, 0.1, 0.01]), k_m=0.5)
brown = Material(vec([0.1, 0.1, 0.1]), k_m=0.5)
gold = Material(vec([.5, .5, 0.1]), k_m=0.5)
snow = Material(vec([1, 1, 1.5]), k_m=0.5)

scene = Scene([
    # Christmas balls - let's make them more shiny/mirror-like
    # Sphere(vec([-4, -0.3, -1]), 0.2, red),
    # Sphere(vec([0.7, 0, 0]), 0.2, red),
    # Sphere(vec([-0.7, 0, 0]), 0.2, green),

    # Snowman :)
    Sphere(vec([1, -0.8, 0]), 0.2, snow),
    Sphere(vec([1, -0.5, 0]), 0.15, snow),
    Sphere(vec([1, -0.3, 0]), 0.1, snow),
    Sphere(vec([1, -0.22, 0]), 0.05, red),
    Sphere(vec([1, -0.275, 0.1]), 0.01, brown),  # my attempt to make eyes
    Sphere(vec([1.05, -0.275, 0.05]), 0.01, brown),

    # Moon
    Sphere(vec([-50, 200, -100]), 25, gold),  # not visible rn :(

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
    AmbientLight(0.005),
]

camera = Camera(vec([3, 1.2, 5]), target=vec(
    [0, 0, 0]), vfov=24, aspect=16/9)

render(camera, scene, lights)
