import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COVER_DIR = os.path.join(BASE_DIR, 'cover_images')
STEGO_DIR = os.path.join(BASE_DIR, 'stego_images')

N_BITS       = 2
EMBED_SEED   = 42

CANNY_LOW    = 100
CANNY_HIGH   = 200
SOBEL_THRESH = 50

IMAGE_SIZE   = (512, 512)

SHORT_MESSAGE   = "Research is the process of discovering a new knowledge."
LONG_MESSAGE    = SHORT_MESSAGE * 70000
MASSIVE_MESSAGE = "TEXT_SECRET_LICENTA " * 50000

IMAGE_NAME_MAP = {
    '4.2.03.tiff'      : 'Baboon',
    '4.2.05.tiff'      : 'F16',
    '4.2.07.tiff'      : 'Peppers',
    '5.2.08.tiff'      : 'Wheel',
    '5.2.10.tiff'      : 'Walkbridge',
    'barbara_gray.tiff': 'Barbara',
    'basket.tiff'      : 'Basket',
    'boat.512.tiff'    : 'Boat',
    'lena_color.tiff'  : 'Lena',
    'livingroom.tiff'  : 'Livingroom',
}