from PIL import Image
import numpy
import matplotlib.pyplot as plt

WORK_FOLDER = "/home/wspek/Code/dev/eko2018/final/whitenoise_test/challenge/files/"


def save_histogram(pil_image, path):
    histogram = pil_image.histogram()

    # plt.figure(0)
    for i in range(0, 256):
        plt.bar(i, histogram[i], color='000', alpha=1.0)

    plt.savefig(path)
    plt.close()


def decompile_gif(file_path):
    image_object = Image.open(file_path)

    # Display individual frames from the loaded animated GIF file
    images = []
    for frame in range(0, image_object.n_frames):
        image_object.seek(frame)
        images.append(image_object.copy())

    return images


def split(images):
    for index, image in enumerate(images):
        # Greyscale the image
        image = image.convert("L")

        numpy_matrix = numpy.asmatrix(image)
        numpy_matrix.setflags(write=1)

        # Get the max value
        max = numpy_matrix.max()

        numpy_matrix_high = numpy.asmatrix(image)
        numpy_matrix_high.setflags(write=1)
        numpy_matrix_high[numpy_matrix < max] = 0

        numpy_matrix_low = numpy.asmatrix(image)
        numpy_matrix_low.setflags(write=1)
        numpy_matrix_low[numpy_matrix == max] = 0

        post_img = Image.fromarray(numpy_matrix_high)
        post_img.save(WORK_FOLDER + 'solution/img_high_{}.png'.format(index))

        post_img = Image.fromarray(numpy_matrix_low)
        post_img.save(WORK_FOLDER + 'solution/img_low_{}.png'.format(index))


def create_complement(prefix_input, prefix_output):
    for image_set in range(0, 5):
        complement = numpy.zeros(shape=(488, 488))  # We can observe that the images have 488 x 488 dimensions

        for num in range(0, 6):
            dir_name = WORK_FOLDER + 'solution'
            file_path = '{}/{}_{}.png'.format(dir_name, prefix_input, ((image_set * 6) + num))

            # For every image, add the matrix representation so we get a complement i.e. the sum of all matrices.
            image_object = Image.open(file_path)
            image_object = image_object.convert("L")    # Convert to greyscale
            matrix = numpy.asmatrix(image_object)

            complement = complement + matrix

        complement_img = Image.fromarray(complement)
        complement_img = complement_img.convert('L')
        complement_img.save(dir_name + '/' + '{}_{}.png'.format(prefix_output, image_set))


def solve():
    # Let's first decompile the GIF to see what we're dealing with
    images = decompile_gif(WORK_FOLDER + "whitenoise.gif")

    # Save the decompiled images
    for index, image in enumerate(images):
        image.save(WORK_FOLDER + 'decompiled/decompiled_{}.png'.format(index), 'PNG')

    # Remove the "subliminal message" from the list. We see it is in index 13
    images.pop(13)

    # For ever image save the histogram
    # for index, image in enumerate(images):
    #     # Greyscale the image
    #     image = image.convert("L")
    #     save_histogram(image, WORK_FOLDER + '/histograms/{}.png'.format(index))

    # The histograms will provide a great hint to the solution. Every histogram contains three pixel colour values:
    # 1) Black , 2) White, 3) White value - 1
    # Let's split every image exactly between the two non-black values and see what we get.
    split(images)

    # By now it should be clear the white noise was encoding QR codes. Also, it should be observed that the
    # "low value" images are the inverse of the "high value" images.
    # Some phones should already be able to decode the high value images. But assuming
    # this is not the case, the contestant needs to observe that the images that have the same QR code actually
    # complement each other. We can get the full original QR codes by:
    create_complement('img_high', 'complement_high')
    create_complement('img_low', 'complement_low')

    # For all the complements of white pixels, print the frequency in which they occur
    for num in range(0, 5):
        dir_name = WORK_FOLDER + 'solution'
        file_path = dir_name + '/' + 'complement_high_{}.png'.format(num)
        image_object = Image.open(file_path)
        histogram = image_object.histogram()
        print histogram[255]

    # Final step: Audacity

if __name__ == '__main__':
    solve()
