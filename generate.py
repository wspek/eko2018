import qrcode
import random
import string
import numpy
import PIL
import imageio
import random
import matplotlib.pyplot as plt

WORK_FOLDER = './static/'


@numpy.vectorize
def binary_invert(current_value, inverted_value):
    if current_value > numpy.uint8(0):
        return 0
    else:
        return numpy.uint8(inverted_value)


@numpy.vectorize
def set_colour(current_value, colour=255):
    if current_value > numpy.uint8(0):
        return colour
    else:
        return current_value


def white_noise(matrix):
    img_mask = numpy.random.randint(0, 2, size=matrix.shape).astype(numpy.bool)
    img_matrix_masked = numpy.ma.masked_array(matrix, mask=img_mask).filled(fill_value=0)

    return img_matrix_masked


def split(matrix, parts=4):
    colour = matrix.max()

    aux_matrix = numpy.zeros(shape=matrix.shape)

    results = []
    for i in range(1, parts):
        white_noised_matrix = white_noise(matrix)
        aux_matrix = aux_matrix + white_noised_matrix
        aux_matrix = set_colour(aux_matrix, colour)

        results.append(white_noised_matrix)

    aux_mask = numpy.matrix(aux_matrix, dtype=bool)
    prerequisite = numpy.ma.masked_array(matrix, mask=aux_mask).filled(fill_value=0)
    white_noised_matrix = white_noise(matrix) + prerequisite
    white_noised_matrix = set_colour(white_noised_matrix, colour)

    results.append(white_noised_matrix)

    return results


def save_histogram(pil_image, path):
    histogram = pil_image.histogram()

    print histogram[255]

    plt.figure(0)
    for i in range(0, 256):
        plt.bar(i, histogram[i], color='000', alpha=1.0)

    plt.savefig(path)


def create_qr_code_set(num=1, colour=255, text=''):
    # Create the QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=8,
        border=18,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color='white', back_color="black").convert('L')

    # DEBUG
    file_name = WORK_FOLDER + 'orig_{}.png'.format(colour)
    img.save(file_name)
    save_histogram(img, WORK_FOLDER + '/' + 'hist_orig.png')

    # Convert the QR code into a numpy matrix
    base_matrix = numpy.asmatrix(img)

    # Set the colour of the image that will result
    base_matrix = set_colour(base_matrix, colour=colour)

    # Get the inverse matrix of the QR matrix and make sure its max colour value is adjusted by one pixel
    base_matrix_inverted = binary_invert(base_matrix, inverted_value=base_matrix.max() - 1)

    # Divide both the base matrix and its inverse in x parts complementary to each other. White noised.
    split_matrices = split(base_matrix, parts=num)
    split_matrices_inverted = split(base_matrix_inverted, parts=num)

    # Add the matrices and their inverses together to obtain a full white noise matrix
    zipped = zip(split_matrices, split_matrices_inverted)
    joined = map(lambda x: x[0] + x[1], zipped)

    return joined


def images2gif(file_names):
    images = []
    for filename in file_names:
        images.append(imageio.imread(filename))

    imageio.mimsave(WORK_FOLDER + 'whitenoise.gif', images, fps=25)


if __name__ == '__main__':
    solution_strings = [
        "UNION",
        "WHITE",
        "PIXELS",
        "FREQUENCY",
        "SALT=SUGAR"
    ]
    longest_length = len(solution_strings[4])

    white_noise_set = []
    for i, solution in enumerate(solution_strings):
        text = solution.ljust(longest_length, ' ')
        white_noise_img = create_qr_code_set(num=6, colour=255, text=text)
        white_noise_set.extend(white_noise_img)

    # N = 1
    # num_sets = 2
    # images_per_set = 6
    # white_noise_set = []
    # # white_noise_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # for i in range(num_sets):
    #     random_string = ''.join(random.choice(string.ascii_letters) for x in range(N))
    #
    #     white_noise_img = create_qr_code_set(num=images_per_set, colour=(255 - i*5), text=random_string)
    #     white_noise_set.extend(white_noise_img)

    file_names = []
    for i, m in enumerate(white_noise_set):
        # added = added + m
        img = PIL.Image.fromarray(m).convert('L')
        save_histogram(img, WORK_FOLDER + 'hist_{}.png'.format(i))

        file_name = WORK_FOLDER + 'result_{}.png'.format(i)
        file_names.append(file_name)
        img.save(file_name)

    # Randomize the file names, so we get a distributed GIF
    # random.shuffle(file_names)

    # Insert subliminal message
    file_names.insert(13, '/home/wspek/Code/dev/eko2018/subliminal/rescue_488.png')

    images2gif(file_names)
