from PIL import Image

im=Image.open("Photon_image.png")
im.rotate(30).show()
im.save("image_rotated.png")