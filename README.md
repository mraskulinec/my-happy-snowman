# My Happy Snowman!
Run our code, which makes use of ray-tracing from scratch, to generate the following fun holiday scene: 

![christmas_scene](https://github.com/mraskulinec/my-happy-snowman/assets/66046109/5554b44e-6adb-466f-b72e-cad6b67db8a8)

The scene includes a snowy ground with a Christmas tree, ornaments, and a snowman under a bright moon. We were originally planning on implementing a cone primitive for the Christmas tree, although it wouldn’t appear on the screen, so we decided to go with a collection of spheres instead. We have a couple of different types of ornaments. Some of them are transparent, which we implemented using refraction. The others were implemented using texture mapping to give them images. We used anti-aliasing (using jittering) to make the edges of objects look smoother. We also added a point light source coming from the moon and the tree topper, which made the lighting of the scene more realistic. Placing the objects in the scene required a bit of trial and error, so to streamline the process we only rendered every fourth pixel, making the rest of them black. Once we had everything where we wanted it, we changed the settings to render every single pixel. This is the final result! 

Feel free to pull the code and make it your own. It's a great way to learn about ray tracing and basic graphics concepts. We are happy to answer any questions!
