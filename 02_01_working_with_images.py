from PIL import Image

im=Image.open("OCR_hindi/page_01_rotated.JPG")
im.rotate(20).show()
im.save("image_rotated.png")