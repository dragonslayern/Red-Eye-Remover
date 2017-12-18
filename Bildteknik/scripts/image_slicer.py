import image_slicer, cv2, sys

images_h = int(sys.argv[1])
images_w = int(sys.argv[2])
path = sys.argv[3]
jpg = ".jpg"
img = cv2.imread(path)
w, h, _ = img.shape
stride_w = w // images_w
stride_h = h // images_h

for i in range(images_w):

	for j in range(images_h):
		cropped_img = img[i*stride_w:(i+1)*stride_w, j*stride_h:(j+1)*stride_h]
		cv2.imwrite(str(path)+str(i)+str(j+1)+str(jpg), cropped_img)