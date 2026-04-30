import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = r'C:\Users\anca\Desktop\Licenta\Articolul 2 - steganografie color\color_blue'
COVER_DIR    = os.path.join(os.path.dirname(BASE_DIR), '..', 'Articolul 1 - steganografie', 'proiect vs-code', 'cover_images')
STEGO_DIR    = os.path.join(BASE_DIR, 'stego_images')

# ── Embedding parameters ──────────────────────────────────────────────────────
N_BITS       = 2
EMBED_SEED   = 42

# ── Edge detection thresholds ────────────────────────────────────────────────
CANNY_LOW    = 100
CANNY_HIGH   = 200
SOBEL_THRESH = 50

# ── Image size ───────────────────────────────────────────────────────────────
IMAGE_SIZE   = (512, 512)

# ── Test messages ─────────────────────────────────────────────────────────────
SHORT_MESSAGE   = "Research is the process of discovering a new knowledge."
LONG_MESSAGE    = SHORT_MESSAGE * 70000
MASSIVE_MESSAGE = "TEXT_SECRET_LICENTA " * 50000

# ── Display names for images ──────────────────────────────────────────────────
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