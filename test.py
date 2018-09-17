import qrcode
import random
import string
import numpy
import PIL
import matplotlib.pyplot as plt

# 1 second
# 24 fps
# 4 shades of grey
# 6 frames per colour


@numpy.vectorize
def binary_invert(current_value, inverted_value):
    if current_value > numpy.uint8(0):
        return 0
    else:
        return numpy.uint8(inverted_value)


@numpy.vectorize
def set_colour(current_value, new_value):
    if current_value > numpy.uint8(0):
        return 0
    else:
        return numpy.uint8(new_value)


def randomize(matrix):
    img_mask = numpy.random.randint(0, 2, size=matrix.shape).astype(numpy.bool)
    img_matrix_masked = numpy.ma.masked_array(matrix, mask=img_mask).filled(fill_value=0)

    return img_matrix_masked


# Create 4 random strings of length n
N = 1
random_string = ''.join(random.choice(string.ascii_letters) for x in range(N))

# Create 4 QR codes

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=20,
    border=0,
)
qr.add_data(random_string)
qr.make(fit=True)

# Create the QR code
pre_img = qr.make_image(fill_color='black', back_color="white").convert('L')
pre_img.save('./static/pre.png')

# Convert the QR code into a matrix
img_matrix = numpy.asmatrix(pre_img)

img_matrix_new_colour = set_colour(img_matrix, new_value=200)
post_img = PIL.Image.fromarray(img_matrix_new_colour)
post_img.save('./static/adjusted.png')

# Get the inverse matrix of the QR matrix and make sure its dark color is one pixel value less
# myfunc_vec = numpy.vectorize(binary_invert)
img_matrix_invert = binary_invert(img_matrix, inverted_value=img_matrix.max()-5)
post_img = PIL.Image.fromarray(img_matrix_invert)
post_img.save('./static/inverted.png')

img_matrix_invert_mask = randomize(img_matrix_invert)
post_img_random = PIL.Image.fromarray(img_matrix_invert_mask)
post_img_random.save('./static/inverted_random.png')

# Create a random img_mask of the same size as the array
img_mask = numpy.random.randint(0, 2, size=img_matrix.shape).astype(numpy.bool)
img_matrix_masked = numpy.ma.masked_array(img_matrix, mask=img_mask).filled(fill_value=0)
post_img_sample = PIL.Image.fromarray(img_matrix_masked)
post_img_sample.save('./static/post_sample.png')

# Both
added_matrix = img_matrix_masked + img_matrix_invert_mask
post_img_sample = PIL.Image.fromarray(added_matrix)
post_img_sample.save('./static/added_random.png')

im = PIL.Image.open('./static/added_random.png')
histogram = im.histogram()

plt.figure(0)
for i in range(0, 256):
    plt.bar(i, histogram[i], color='000', alpha=1.0)
plt.show()
