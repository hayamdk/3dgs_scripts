from PIL import Image, ImageDraw
import math
import sys

mask_path = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
cx = float(sys.argv[4])
cy = float(sys.argv[5])
r = float(sys.argv[6])

image = Image.new('RGB', (width, height), (0, 0, 0))
draw = ImageDraw.Draw(image)

for v in range(0, height):
	print("Line", v, "of", height)
	for u in range(0, width):
		u_norm = u / width * 2 - cx
		v_norm = v / height * 2 - cy

		if math.sqrt(u_norm * u_norm + v_norm * v_norm) < r:
			draw.point((u, v), fill=(255, 255, 255))
		else:
			draw.point((u, v), fill=(0, 0, 0))

image.save(mask_path)
